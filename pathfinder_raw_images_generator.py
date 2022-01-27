import argparse
import os
import random
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

# Based on the original script from:
# "Learning long-range spatial dependencies with horizontal gated-recurrent units" (Linsley et al. 2018) <https://arxiv.org/abs/1805.08315>

def accumulate_meta(array, label, subpath, filename, args, nimg, paddle_margin=None):
    array += [[subpath, filename, nimg, label,
               args.continuity, args.contour_length, args.distractor_length,
               args.paddle_length, args.paddle_thickness, paddle_margin, len(args.paddle_contrast_list)]]
    return array


def two_snakes(image_size, padding, seed_distance,
               num_segments, segment_length, thickness, margin, continuity, small_dilation_structs,
               large_dilation_structs,
               snake_contrast_list,
               paddle_contrast_list,
               max_segment_trial, aa_scale,
               display_snake=False, display_segment=False,
               allow_shorter_snakes=False):
    # sample contrast centers of two snakes
    snake_contrast_mu_list = snake_contrast_list * 2
    random.shuffle(snake_contrast_mu_list)
    snake_contrast_mu_list = snake_contrast_mu_list[:2]

    # draw initial segment
    num_possible_contrasts = len(paddle_contrast_list)
    # for isegment in range(1):
    current_images, current_mask, current_segment_masks, current_pivots, current_orientations, origin_tips, success \
        = initialize_two_seeds(image_size, padding, seed_distance,
                               segment_length, thickness, margin, snake_contrast_mu_list, paddle_contrast_list,
                               large_dilation_structs,
                               max_segment_trial,
                               aa_scale, display=display_segment)
    if success is False:
        return np.zeros((image_size[0], image_size[1])), np.zeros((image_size[0], image_size[1])), None, None, False

    # sequentially add segments
    terminal_tips = [[0, 0], [0, 0]]
    for isegment in range(num_segments - 1):
        if num_possible_contrasts > 0:
            contrast_index = np.random.randint(low=0, high=num_possible_contrasts)
        else:
            contrast_index = 0
        contrast = paddle_contrast_list[contrast_index]
        for isnake in range(len(current_segment_masks)):
            current_images[isnake], current_mask, current_segment_masks[isnake], current_pivots[isnake], \
            current_orientations[isnake], terminal_tips[isnake], success \
                = extend_snake(list(current_pivots[isnake]), current_orientations[isnake],
                               current_segment_masks[isnake],
                               current_images[isnake], current_mask, max_segment_trial,
                               segment_length, thickness, margin, continuity, contrast * snake_contrast_mu_list[isnake],
                               small_dilation_structs, large_dilation_structs,
                               aa_scale=aa_scale,
                               display=display_segment,
                               forced_current_pivot=None)
            if success is False:
                if allow_shorter_snakes:
                    return current_images, current_mask, None, None, True
                else:
                    return current_images, current_mask, None, None, False
    current_mask = np.maximum(current_mask, current_segment_masks[-1])
    # display snake
    if display_snake:
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(np.maximum(current_images[0], current_images[1]))
        plt.subplot(1, 2, 2)
        plt.imshow(current_mask)
        plt.show()
    return current_images, current_mask, origin_tips, terminal_tips, True


def initialize_two_seeds(image_size, padding, seed_distance,
                         length, thickness, margin, snakes_contrast_mu_list, paddle_contrast_list,
                         large_dilation_structs,
                         max_segment_trial,
                         aa_scale, display=False):
    image1 = np.zeros((image_size[0], image_size[1]))
    image2 = np.zeros((image_size[0], image_size[1]))
    mask = np.zeros((image_size[0], image_size[1]))
    mask[:padding, :] = 1
    mask[-padding:, :] = 1
    mask[:, :padding] = 1
    mask[:, -padding:] = 1

    struct_shape = ((length + margin) * 2 + 1, (length + margin) * 2 + 1)
    struct_head = [length + margin + 1, length + margin + 1]

    # SAMPLE FIRST SEGMENT
    num_possible_contrasts = len(paddle_contrast_list)
    if num_possible_contrasts > 1:
        contrast_index = np.random.randint(low=0, high=num_possible_contrasts)
    else:
        contrast_index = 0
    contrast = paddle_contrast_list[contrast_index] * snakes_contrast_mu_list[0]
    trial_count = 0
    while trial_count <= max_segment_trial:
        sampled_orientation_in_rad1 = np.random.randint(low=-180, high=180) * np.pi / 180
        if sampled_orientation_in_rad1 + np.pi < np.pi:
            sampled_orientation_in_rad_reversed = sampled_orientation_in_rad1 + np.pi
        else:
            sampled_orientation_in_rad_reversed = sampled_orientation_in_rad1 - np.pi

        # generate dilation struct
        _, struct = draw_line_n_mask(struct_shape, struct_head, sampled_orientation_in_rad1, length, thickness, margin,
                                     large_dilation_structs, aa_scale)
        # head-centric struct

        # dilate mask using segment
        lined_mask = mask.copy()
        lined_mask[:seed_distance * 2, :] = 1
        lined_mask[image_size[0] - seed_distance * 2:, :] = 1
        lined_mask[:, :seed_distance * 2] = 1
        lined_mask[:, image_size[1] - seed_distance * 2:] = 1
        dilated_mask = binary_dilate_custom(lined_mask, struct, value_scale=1.)
        # dilation in the same orientation as the tail

        # run coordinate searcher while also further dilating
        _, raw_num_available_coordinates = find_available_coordinates(np.ceil(mask - 0.3), margin=0)
        available_coordinates, num_available_coordinates = find_available_coordinates(np.ceil(dilated_mask - 0.3),
                                                                                      margin=0)
        if num_available_coordinates == 0:
            # print('Mask fully occupied after dilation. finalizing')
            return image1, mask, [np.zeros_like(mask), np.zeros_like(mask)], [None, None], [None, None], [None,
                                                                                                          None], False

        # sample coordinate and draw
        random_number = np.random.randint(low=0, high=num_available_coordinates)
        sampled_tail1 = [available_coordinates[0][random_number],
                         available_coordinates[1][random_number]]  # CHECK OUT OF BOUNDARY CASES
        sampled_head1 = translate_coord(sampled_tail1, sampled_orientation_in_rad1, length)
        sampled_pivot1 = translate_coord(sampled_head1, sampled_orientation_in_rad_reversed, length + margin)
        sampled_tip1 = [sampled_tail1[0], sampled_tail1[1]]
        if (sampled_head1[0] < 0) | (sampled_head1[0] >= mask.shape[0]) | \
                (sampled_head1[1] < 0) | (sampled_head1[1] >= mask.shape[1]) | \
                (sampled_pivot1[0] < 0) | (sampled_pivot1[0] >= mask.shape[0]) | \
                (sampled_pivot1[1] < 0) | (sampled_pivot1[1] >= mask.shape[1]):
            # print('missampled seed +segment_trial_count')
            trial_count += 1
            continue
        else:
            break
    if trial_count > max_segment_trial:
        return image1, mask, [np.zeros_like(mask), np.zeros_like(mask)], [None, None], [None, None], [None, None], False
    l_im, m_im1 = draw_line_n_mask((mask.shape[0], mask.shape[1]), sampled_tail1, sampled_orientation_in_rad1, length,
                                   thickness, margin, large_dilation_structs, aa_scale, contrast_scale=contrast)
    image1 = np.maximum(image1, l_im)

    # SAMPLE SECOND SEGMENT
    num_possible_contrasts = len(paddle_contrast_list)
    if num_possible_contrasts > 1:
        contrast_index = np.random.randint(low=0, high=num_possible_contrasts)
    else:
        contrast_index = 0
    contrast = paddle_contrast_list[contrast_index] * snakes_contrast_mu_list[1]
    trial_count = 0
    while trial_count <= max_segment_trial:
        sampled_orientation_in_rad2 = np.random.randint(low=-180, high=180) * np.pi / 180
        if sampled_orientation_in_rad2 + np.pi < np.pi:
            sampled_orientation_in_rad_reversed = sampled_orientation_in_rad2 + np.pi
        else:
            sampled_orientation_in_rad_reversed = sampled_orientation_in_rad2 - np.pi

        sample_in_rad = np.random.randint(0, 360) * np.pi / 180
        # get lists of y and x coordinates (exclude out-of-bound coordinates)
        sample_in_y = int(np.round_(sampled_tail1[0] + (seed_distance * np.sin(sample_in_rad))))
        sample_in_x = int(np.round_(sampled_tail1[1] + (seed_distance * np.cos(sample_in_rad))))
        sampled_tail2 = [sample_in_y, sample_in_x]
        sampled_head2 = translate_coord(sampled_tail2, sampled_orientation_in_rad2, length)
        sampled_pivot2 = translate_coord(sampled_head2, sampled_orientation_in_rad_reversed, length + margin)
        sampled_tip2 = [sampled_tail2[0], sampled_tail2[1]]
        if (sampled_head2[0] < 0) | (sampled_head2[0] >= mask.shape[0]) | \
                (sampled_head2[1] < 0) | (sampled_head2[1] >= mask.shape[1]) | \
                (sampled_pivot2[0] < 0) | (sampled_pivot2[0] >= mask.shape[0]) | \
                (sampled_pivot2[1] < 0) | (sampled_pivot2[1] >= mask.shape[1]):
            # print('missampled seed +segment_trial_count')
            trial_count += 1
            continue
        else:
            break
    if trial_count > max_segment_trial:
        return image2, mask, [np.zeros_like(mask), np.zeros_like(mask)], [None, None], [None, None], [None, None], False

    l_im, m_im2 = draw_line_n_mask((mask.shape[0], mask.shape[1]), sampled_tail2, sampled_orientation_in_rad2, length,
                                   thickness, margin, large_dilation_structs, aa_scale, contrast_scale=contrast)
    image2 = np.maximum(image2, l_im)

    if display:
        plt.figure(figsize=(10, 20))
        plt.imshow(np.maximum(image1, image2))
        plt.title(str(num_available_coordinates))
        plt.plot(sampled_tail1[1], sampled_tail1[0], 'bo')
        plt.plot(sampled_head1[1], sampled_head1[0], 'ro')
        plt.plot(sampled_tail2[1], sampled_tail2[0], 'bo')
        plt.plot(sampled_head2[1], sampled_head2[0], 'ro')
        plt.show()

    return [image1, image2], mask, [m_im1, m_im2], \
           [sampled_pivot1, sampled_pivot2], \
           [sampled_orientation_in_rad1, sampled_orientation_in_rad2], \
           [sampled_tip1, sampled_tip2], True


def draw_circle(window_size, coordinate, radius, aa_scale):
    image = np.zeros((window_size[0] * aa_scale, window_size[1] * aa_scale))
    y, x = np.ogrid[-coordinate[0] * aa_scale:(window_size[0] - coordinate[0]) * aa_scale,
           -coordinate[1] * aa_scale:(window_size[1] - coordinate[1]) * aa_scale]
    mask = x ** 2 + y ** 2 <= (radius * aa_scale) ** 2
    image[mask] = 1
    return np.array(
        Image.fromarray(np.uint8(image * 255)).resize(size=(window_size[0], window_size[1]), resample=Image.LANCZOS))


def save_metadata(metadata, contour_path):
    # Converts metadata (list of lists) into an nparray, and then saves
    if not os.path.exists(contour_path):
        os.makedirs(contour_path)
    metadata_fn = 'metadata.npy'
    np.save(os.path.join(contour_path, metadata_fn), metadata)


def make_many_snakes(image, mask,
                     num_snakes, max_snake_trial,
                     num_segments, segment_length, thickness, margin, continuity,
                     contrast_list,
                     max_segment_trial,
                     aa_scale,
                     display_final=False, display_snake=False, display_segment=False,
                     allow_incomplete=False,
                     allow_shorter_snakes=False,
                     stop_with_availability=None):
    curr_image = image.copy()
    curr_mask = mask.copy()
    isnake = 0

    small_dilation_structs = generate_dilation_struct(margin)
    large_dilation_structs = generate_dilation_struct(margin * aa_scale)

    if image is None:
        print('No image. Previous run probably failed.')
    while isnake < num_snakes:
        snake_retry_count = 0
        while snake_retry_count <= max_snake_trial:
            curr_image, curr_mask, success = \
                make_snake(curr_image, curr_mask,
                           num_segments, segment_length, thickness, margin, continuity, small_dilation_structs,
                           large_dilation_structs,
                           contrast_list,
                           max_segment_trial,
                           aa_scale, display_snake, display_segment, allow_shorter_snakes, stop_with_availability)
            if success is False:
                snake_retry_count += 1
            else:
                break
        if snake_retry_count > max_snake_trial:
            # print('Exceeded max # of snake re-rendering.')
            if not allow_incomplete:
                print('Required # snakes unmet. Aborting')
                return None, None
        isnake += 1
    if display_final:
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(curr_image)
        plt.subplot(1, 2, 2)
        plt.imshow(curr_mask)
        plt.show()
    return curr_image, curr_mask


def find_available_coordinates(mask, margin):
    if np.min(mask) < 0:
        # print('mystery')
        return (np.array([]), np.array([])), 0
    # get temporarily dilated mask
    if margin > 0:
        dilated_mask = mask.copy()
        # dilated_mask = binary_dilate(mask, margin, type='1', scale=1.)

    elif margin == 0:
        dilated_mask = mask.copy()
    # get a list of available coordinates
    available_coordinates = np.nonzero(1 - dilated_mask.astype(np.uint8))
    num_available_coordinates = available_coordinates[0].shape[0]
    return available_coordinates, num_available_coordinates


def make_snake(image, mask,
               num_segments, segment_length, thickness, margin, continuity, small_dilation_structs,
               large_dilation_structs,
               contrast_list,
               max_segment_trial, aa_scale,
               display_snake=False, display_segment=False,
               allow_shorter_snakes=False, stop_with_availability=None):
    # set recurring state variables
    current_segment_mask = np.zeros_like(mask)
    current_image = image.copy()
    current_mask = mask.copy()
    # draw initial segment
    for isegment in range(1):
        num_possible_contrasts = len(contrast_list)
        if num_possible_contrasts > 0:
            contrast_index = np.random.randint(low=0, high=num_possible_contrasts)
        else:
            contrast_index = 0
        contrast = contrast_list[contrast_index]
        current_image, current_mask, current_segment_mask, current_pivot, current_orientation, success \
            = seed_snake(current_image, current_mask,
                         max_segment_trial, segment_length, thickness, margin, contrast, small_dilation_structs,
                         large_dilation_structs,
                         aa_scale=aa_scale, display=display_segment, stop_with_availability=stop_with_availability)
        if success is False:
            return image, mask, False
    # sequentially add segments
    for isegment in range(int(num_segments) - 1):
        if num_possible_contrasts > 0:
            contrast_index = np.random.randint(low=0, high=num_possible_contrasts)
        else:
            contrast_index = 0
        contrast = contrast_list[contrast_index]
        current_image, current_mask, current_segment_mask, current_pivot, current_orientation, _, success \
            = extend_snake(list(current_pivot), current_orientation, current_segment_mask,
                           current_image, current_mask, max_segment_trial,
                           segment_length, thickness, margin, continuity, contrast, small_dilation_structs,
                           large_dilation_structs,
                           aa_scale=aa_scale,
                           display=display_segment,
                           forced_current_pivot=None)
        if success is False:
            if allow_shorter_snakes:
                return image, mask, True
            else:
                return image, mask, False
    current_mask = np.maximum(current_mask, current_segment_mask)
    # display snake
    if display_snake:
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(current_image)
        plt.subplot(1, 2, 2)
        plt.imshow(current_mask)
        plt.show()
    return current_image, current_mask, True


def seed_snake(image, mask,
               max_segment_trial, length, thickness, margin, contrast, small_dilation_structs, large_dilation_structs,
               aa_scale, display=False, stop_with_availability=None):
    struct_shape = ((length + margin) * 2 + 1, (length + margin) * 2 + 1)
    struct_head = [length + margin + 1, length + margin + 1]

    trial_count = 0
    while trial_count <= max_segment_trial:
        sampled_orientation_in_rad = np.random.randint(low=-180, high=180) * np.pi / 180
        if sampled_orientation_in_rad + np.pi < np.pi:
            sampled_orientation_in_rad_reversed = sampled_orientation_in_rad + np.pi
        else:
            sampled_orientation_in_rad_reversed = sampled_orientation_in_rad - np.pi

        # generate dilation struct
        _, struct = draw_line_n_mask(struct_shape, struct_head, sampled_orientation_in_rad, length, thickness, margin,
                                     large_dilation_structs, aa_scale)
        # head-centric struct

        # dilate mask using segment
        lined_mask = mask.copy()
        lined_mask[0, :] = 1
        lined_mask[-1, :] = 1
        lined_mask[:, 0] = 1
        lined_mask[:, -1] = 1
        dilated_mask = binary_dilate_custom(lined_mask, struct, value_scale=1.)
        # dilation in the same orientation as the tail

        # run coordinate searcher while also further dilating
        _, raw_num_available_coordinates = find_available_coordinates(np.ceil(mask - 0.3), margin=0)
        available_coordinates, num_available_coordinates = find_available_coordinates(np.ceil(dilated_mask - 0.3),
                                                                                      margin=0)
        if (stop_with_availability is not None) & \
                (np.float64(raw_num_available_coordinates) / (mask.shape[0] * mask.shape[1]) < stop_with_availability):
            # print('critical % of mask occupied before dilation. finalizing')
            return image, mask, np.zeros_like(mask), None, None, False
        if num_available_coordinates == 0:
            # print('Mask fully occupied after dilation. finalizing')
            return image, mask, np.zeros_like(mask), None, None, False
            continue

        # sample coordinate and draw
        random_number = np.random.randint(low=0, high=num_available_coordinates)
        sampled_tail = [available_coordinates[0][random_number],
                        available_coordinates[1][random_number]]  # CHECK OUT OF BOUNDARY CASES
        sampled_head = translate_coord(sampled_tail, sampled_orientation_in_rad, length)
        sampled_pivot = translate_coord(sampled_head, sampled_orientation_in_rad_reversed, length + margin)

        if (sampled_head[0] < 0) | (sampled_head[0] >= mask.shape[0]) | \
                (sampled_head[1] < 0) | (sampled_head[1] >= mask.shape[1]) | \
                (sampled_pivot[0] < 0) | (sampled_pivot[0] >= mask.shape[0]) | \
                (sampled_pivot[1] < 0) | (sampled_pivot[1] >= mask.shape[1]):
            trial_count += 1
            continue
        else:
            break
    if trial_count > max_segment_trial:
        return image, mask, np.zeros_like(mask), None, None, False

    l_im, m_im = draw_line_n_mask((mask.shape[0], mask.shape[1]), sampled_tail, sampled_orientation_in_rad, length,
                                  thickness, margin, large_dilation_structs, aa_scale, contrast_scale=contrast)
    image = np.maximum(image, l_im)

    if display:
        plt.figure(figsize=(10, 20))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.subplot(1, 3, 2)
        plt.imshow(mask)
        plt.subplot(1, 3, 3)
        plt.imshow(dilated_mask)
        plt.title(str(num_available_coordinates))
        plt.plot(sampled_tail[1], sampled_tail[0], 'bo')
        plt.plot(sampled_head[1], sampled_head[0], 'ro')
        plt.show()
    return image, mask, m_im, sampled_pivot, sampled_orientation_in_rad, True


def extend_snake(last_pivot, last_orientation, last_segment_mask,
                 image, mask, max_segment_trial,
                 length, thickness, margin, continuity, contrast, small_dilation_structs, large_dilation_structs,
                 aa_scale,
                 display=False,
                 forced_current_pivot=None):
    # set anchor
    if forced_current_pivot is not None:
        new_pivot = list(forced_current_pivot)
    else:
        new_pivot = translate_coord(last_pivot, last_orientation, length + 2 * margin)
    # get temporarily dilated mask
    dilated_mask = binary_dilate_custom(mask, small_dilation_structs, value_scale=1.)
    # get candidate endpoints
    unique_coords, unique_orientations, cmf, pmf = get_coords_cmf(new_pivot, last_orientation, length + margin,
                                                                  dilated_mask, continuity)
    # sample endpoint
    if pmf is None:
        return image, mask, None, None, None, None, False
    else:
        trial_count = 0
        segment_found = False
        while trial_count <= max_segment_trial:
            random_num = np.random.rand()
            sampled_index = np.argmax(cmf - random_num > 0)
            new_orientation = unique_orientations[sampled_index]
            new_head = unique_coords[sampled_index, :]  # find the smallest index whose value is greater than rand
            flipped_orientation = flip_by_pi(new_orientation)
            l_im, m_im = draw_line_n_mask((mask.shape[0], mask.shape[1]), new_head, flipped_orientation, length,
                                          thickness, margin, large_dilation_structs, aa_scale, contrast_scale=contrast)

            trial_count += 1
            if np.max(mask + m_im) < 1.8:
                segment_found = True
                break
            else:
                continue

        if segment_found == False:
            print('extend_snake: self-crossing detected')
            print('pmf of sample =' + str(pmf[0, sampled_index]))
            print('mask at sample =' + str(dilated_mask[new_head[0], new_head[1]]))
            print('smaple =' + str(new_head))
            image = np.maximum(image, l_im)
            if display:
                plt.subplot(1, 4, 1)
                plt.imshow(image)
                plt.subplot(1, 4, 2)
                plt.imshow(mask)
                plt.subplot(1, 4, 3)
                plt.imshow(dilated_mask)
                plt.subplot(1, 4, 4)
                plt.imshow(mask + m_im)
                plt.plot(new_pivot[1], new_pivot[0], 'go')
                plt.plot(new_head[1], new_head[0], 'ro')
                plt.show()
            return image, mask, None, None, None, None, False
        else:
            image = np.maximum(image, l_im)
            mask = np.maximum(mask, last_segment_mask)
            if display:
                plt.subplot(1, 2, 1)
                plt.imshow(image)
                plt.subplot(1, 2, 2)
                plt.imshow(mask)
                plt.plot(new_pivot[1], new_pivot[0], 'go')
                plt.plot(new_head[1], new_head[0], 'ro')
                plt.show()
            return image, mask, m_im, new_pivot, new_orientation, new_head, True


def get_coords_cmf(last_endpoint, last_orientation, step_length, mask, continuity):
    height = mask.shape[0]
    width = mask.shape[1]
    # compute angle of an arc whose length equals to the size of one pixel
    deg_per_pixel = 360. / (2 * step_length * np.pi)
    samples_in_rad = np.arange(0, 360, deg_per_pixel) * np.pi / 180
    # get lists of y and x coordinates (exclude out-of-bound coordinates)
    samples_in_y = last_endpoint[0] + (step_length * np.sin(samples_in_rad))
    samples_in_x = last_endpoint[1] + (step_length * np.cos(samples_in_rad))
    samples_in_coord = np.concatenate((np.expand_dims(samples_in_y, axis=1),
                                       np.expand_dims(samples_in_x, axis=1)), axis=1).astype(int)
    OOB_rows = (samples_in_y >= height) | (samples_in_y < 0) | (samples_in_x >= width) | (samples_in_x < 0)
    samples_in_coord = np.delete(samples_in_coord, np.where(OOB_rows), axis=0)
    if samples_in_coord.shape[0] == 0:
        # print('dead-end while expanding')
        return None, None, None, None
    # find unique coordinates and related quantities
    unique_coords, indices = np.unique(samples_in_coord, axis=0, return_index=True)
    unique_displacements = unique_coords - last_endpoint
    unique_orientations = np.arctan2(unique_displacements[:, 0], unique_displacements[:, 1])  # from -pi to pi
    unique_delta = np.minimum(np.abs(unique_orientations - last_orientation),
                              2 * np.pi - np.abs(unique_orientations - last_orientation))
    unique_delta = np.minimum(unique_delta * continuity,
                              0.5 * np.pi)
    unique_cosinedistweights = np.maximum(np.cos(unique_delta), 0) ** 2
    # compute probability distribution
    inverted = 1 - mask[unique_coords[:, 0], unique_coords[:, 1]]
    pmf = np.multiply(np.array([inverted]), unique_cosinedistweights)  # weight coordinates according to continuity
    total = np.sum(pmf)
    if total < 1e-4:
        # print('dead-end while expanding')
        return None, None, None, None
    pmf = pmf / total
    cmf = np.cumsum(pmf)
    return unique_coords, unique_orientations, cmf, pmf


def draw_line_n_mask(im_size, start_coord, orientation, length, thickness, margin, large_dilation_struct, aa_scale,
                     contrast_scale=1.0):
    # sanity check
    if np.round(thickness * aa_scale) - thickness * aa_scale != 0.0:
        raise ValueError('thickness does not break even.')

    # draw a line in a finer resolution
    miniline_blown_shape = (length + int(np.ceil(thickness)) + margin) * 2 * aa_scale + 1
    miniline_blown_center = (length + int(np.ceil(thickness)) + margin) * aa_scale
    miniline_blown_thickness = int(np.round(thickness * aa_scale))
    miniline_blown_head = translate_coord([miniline_blown_center, miniline_blown_center], orientation,
                                          length * aa_scale)
    miniline_blown_im = Image.new('F', (miniline_blown_shape, miniline_blown_shape), 'black')
    line_draw = ImageDraw.Draw(miniline_blown_im)
    line_draw.line([(miniline_blown_center, miniline_blown_center),
                    (miniline_blown_head[1], miniline_blown_head[0])],
                   fill='white', width=miniline_blown_thickness)

    # resize with interpolation + apply contrast
    miniline_shape = (length + int(np.ceil(thickness)) + margin) * 2 + 1
    miniline_im = np.array(miniline_blown_im.resize(size=(miniline_shape, miniline_shape),
                                                    resample=Image.LANCZOS)).astype(float) / 255
    if contrast_scale != 1.0:
        miniline_im *= contrast_scale

    # draw a mask
    minimask_blown_im = binary_dilate_custom(miniline_blown_im, large_dilation_struct, value_scale=1.).astype(np.uint8)
    minimask_blown_im = Image.fromarray(minimask_blown_im)
    minimask_im = np.array(minimask_blown_im.resize(size=(miniline_shape, miniline_shape),
                                                    resample=Image.LANCZOS)).astype(float) / 255

    # place in original shape
    l_im = np.array(Image.new('F', (im_size[1], im_size[0]), 'black'))
    m_im = l_im.copy()
    l_im_vertical_range_raw = [start_coord[0] - (length + int(np.ceil(thickness)) + margin),
                               start_coord[0] + (length + int(np.ceil(thickness)) + margin)]
    l_im_horizontal_range_raw = [start_coord[1] - (length + int(np.ceil(thickness)) + margin),
                                 start_coord[1] + (length + int(np.ceil(thickness)) + margin)]
    l_im_vertical_range_rectified = [np.maximum(l_im_vertical_range_raw[0], 0),
                                     np.minimum(l_im_vertical_range_raw[1], im_size[0] - 1)]
    l_im_horizontal_range_rectified = [np.maximum(l_im_horizontal_range_raw[0], 0),
                                       np.minimum(l_im_horizontal_range_raw[1], im_size[1] - 1)]
    miniline_im_vertical_range_rectified = [np.maximum(0, -l_im_vertical_range_raw[0]),
                                            miniline_shape - 1 - np.maximum(0, l_im_vertical_range_raw[1] - (
                                                    im_size[0] - 1))]
    miniline_im_horizontal_range_rectified = [np.maximum(0, -l_im_horizontal_range_raw[0]),
                                              miniline_shape - 1 - np.maximum(0, l_im_horizontal_range_raw[1] - (
                                                      im_size[1] - 1))]
    l_im[l_im_vertical_range_rectified[0]:l_im_vertical_range_rectified[1] + 1,
    l_im_horizontal_range_rectified[0]:l_im_horizontal_range_rectified[1] + 1] = \
        miniline_im[miniline_im_vertical_range_rectified[0]:miniline_im_vertical_range_rectified[1] + 1,
        miniline_im_horizontal_range_rectified[0]:miniline_im_horizontal_range_rectified[1] + 1].copy()
    m_im[l_im_vertical_range_rectified[0]:l_im_vertical_range_rectified[1] + 1,
    l_im_horizontal_range_rectified[0]:l_im_horizontal_range_rectified[1] + 1] = \
        minimask_im[miniline_im_vertical_range_rectified[0]:miniline_im_vertical_range_rectified[1] + 1,
        miniline_im_horizontal_range_rectified[0]:miniline_im_horizontal_range_rectified[1] + 1].copy()

    return l_im, m_im


def binary_dilate_custom(im, struct, value_scale=1.):
    # out = ndimage.morphology.binary_dilation(np.array(im), structure=struct, iterations=iterations)
    out = np.array(cv2.dilate(np.array(im), kernel=struct.astype(np.uint8), iterations=1)).astype(float) / value_scale
    # out = np.minimum(signal.fftconvolve(np.array(im), struct, mode='same').astype(np.uint8), np.ones_like(im))
    return out


def generate_dilation_struct(margin):
    kernel = np.zeros((2 * margin + 1, 2 * margin + 1))
    y, x = np.ogrid[-margin:margin + 1, -margin:margin + 1]
    mask = x ** 2 + y ** 2 <= margin ** 2
    kernel[mask] = 1
    return kernel


def translate_coord(coord, orientation, dist, allow_float=False):
    y_displacement = float(dist) * np.sin(orientation)
    x_displacement = float(dist) * np.cos(orientation)
    if allow_float is True:
        new_coord = [coord[0] + y_displacement, coord[1] + x_displacement]
    else:
        new_coord = [int(np.ceil(coord[0] + y_displacement)), int(np.ceil(coord[1] + x_displacement))]
    return new_coord


def flip_by_pi(orientation):
    if orientation < 0:
        flipped_orientation = orientation + np.pi
    else:
        flipped_orientation = orientation - np.pi
    return flipped_orientation


def from_wrapper(args):
    t = time.time()
    iimg = 0

    contour_sub_path = os.path.join('imgs')
    if not os.path.exists(os.path.join(args.contour_path, contour_sub_path)):
        os.makedirs(os.path.join(args.contour_path, contour_sub_path))

    metadata = []
    # CHECK IF METADATA FILE ALREADY EXISTS
    metadata_path = args.contour_path
    if not os.path.exists(metadata_path):
        os.makedirs(metadata_path)
    metadata_fn = 'metadata.npy'
    metadata_full = os.path.join(metadata_path, metadata_fn)
    if os.path.exists(metadata_full):
        print('Metadata file already exists.')
        return

    while iimg < args.num_samples:
        label = np.random.randint(low=0, high=2)
        print('Image# : %s' % (iimg))

        # Sample paddle margin
        num_possible_margins = len(args.paddle_margin_list)
        if num_possible_margins > 0:
            margin_index = np.random.randint(low=0, high=num_possible_margins)
        else:
            margin_index = 0

        margin = args.paddle_margin_list[margin_index]
        base_num_paddles = 150
        num_paddles_factor = 1. / ((7.5 + 13 * margin + 4 * margin * margin) / 123.5)
        total_num_paddles = int(base_num_paddles * num_paddles_factor)

        small_dilation_structs = generate_dilation_struct(margin)
        large_dilation_structs = generate_dilation_struct(margin * args.antialias_scale)

        ### SAMPLE TWO TARGET SNAKES
        success = False
        twosnakes = mask = None
        origin_tips = terminal_tips = None
        while not success:
            twosnakes, mask, origin_tips, terminal_tips, success = \
                two_snakes(args.window_size, args.padding, args.seed_distance,
                           args.contour_length, args.paddle_length, args.paddle_thickness, margin, args.continuity,
                           small_dilation_structs, large_dilation_structs,
                           args.snake_contrast_list,
                           args.paddle_contrast_list,
                           args.max_paddle_retrial,
                           args.antialias_scale,
                           display_snake=False, display_segment=False,
                           allow_shorter_snakes=False)

        image = np.maximum(twosnakes[0], twosnakes[1])
        ### SAMPLE SHORT SNAKE DISTRACTORS
        num_distractor_snakes = args.num_distractor_snakes
        if num_distractor_snakes > 0:
            image, mask = make_many_snakes(image, mask,
                                           num_distractor_snakes, args.max_distractor_contour_retrial,
                                           args.distractor_length, args.paddle_length, args.paddle_thickness, margin,
                                           args.continuity,
                                           args.snake_contrast_list,
                                           args.max_paddle_retrial,
                                           args.antialias_scale,
                                           display_final=False, display_snake=False, display_segment=False,
                                           allow_incomplete=True, allow_shorter_snakes=False,
                                           stop_with_availability=0.01)

        if image is None:
            continue
        if args.use_single_paddles is not False:
            ### SAMPLE SINGLE PADDLE DISTRACTORS
            num_single_paddles = total_num_paddles - 2 * args.contour_length - num_distractor_snakes * args.distractor_length
            image, _ = make_many_snakes(image, mask,
                                        num_single_paddles, args.max_paddle_retrial,
                                        1, args.paddle_length, args.paddle_thickness, margin, args.continuity,
                                        args.snake_contrast_list,
                                        args.max_paddle_retrial,
                                        args.antialias_scale,
                                        display_final=False, display_snake=False, display_segment=False,
                                        allow_incomplete=True, allow_shorter_snakes=False,
                                        stop_with_availability=0.01)
            if image is None:
                continue

        ### ADD MARKERS
        origin_mark_idx = np.random.randint(0, 2)
        if label == 0:
            terminal_mark_idx = 1 - origin_mark_idx
        else:
            terminal_mark_idx = origin_mark_idx
        origin_mark_coord = origin_tips[origin_mark_idx]
        terminal_mark_coord = terminal_tips[terminal_mark_idx]
        origin_circle = draw_circle(args.window_size, origin_mark_coord, args.marker_radius, args.antialias_scale)
        terminal_circle = draw_circle(args.window_size, terminal_mark_coord, args.marker_radius, args.antialias_scale)

        markers = np.maximum(origin_circle, terminal_circle).astype(float) / 255
        image_marked = np.maximum(image, markers)

        fn = "sample_%s.png" % (iimg)
        image_marked /= image_marked.max()
        Image.fromarray(image_marked * 255).convert('L').save(os.path.join(args.contour_path, contour_sub_path, fn))
        metadata = accumulate_meta(metadata, label, contour_sub_path, fn, args, iimg, paddle_margin=margin)

        iimg += 1

    matadata_nparray = np.array(metadata)
    save_metadata(matadata_nparray, args.contour_path)

    elapsed = time.time() - t
    print('ELAPSED TIME : ', str(elapsed))

    return


def main():
    t = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="data/pathfinder/raw",
                        help="Path to the directory to save the datasets at.")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to generate.")
    parser.add_argument("--path_lengths", nargs="+", type=int, default=[3, 5, 7, 9],
                        help="Path lengths to generate raw data for (one dataset for each path length).")

    parser.add_argument("--window_size", nargs="+", type=int, default=[300, 300], help="Generated image size.")
    parser.add_argument("--padding", type=int, default=22,
                        help="Padding argument corresponding to the argument mentioned in https://arxiv.org/pdf/1805.08315.pdf")
    parser.add_argument("--continuity", type=float, default=1.8,
                        help="Continuity argument corresponding to the argument mentioned in https://arxiv.org/pdf/1805.08315.pdf")
    parser.add_argument("--antialias_scale", type=int, default=2, help="")
    parser.add_argument("--snake_contrast_list", nargs="+", type=float, default=[1.0], help="")
    parser.add_argument("--marker_radius", type=int, default=3,
                        help="Marker radius argument corresponding to the argument mentioned in https://arxiv.org/pdf/1805.08315.pdf")
    parser.add_argument("--max_distractor_contour_retrial", type=int, default=4, help="")
    parser.add_argument("--use_single_paddles", type=bool, default=False, help="")
    parser.add_argument("--seed_distance", type=int, default=27, help="")

    parser.add_argument("--paddle_length", type=int, default=5,
                        help="Paddle length radius argument corresponding to the argument mentioned in https://arxiv.org/pdf/1805.08315.pdf")
    parser.add_argument("--paddle_thickness", type=int, default=2,
                        help="Paddle thickness radius argument corresponding to the argument mentioned in https://arxiv.org/pdf/1805.08315.pdf")
    parser.add_argument("--paddle_contrast_list", nargs="+", type=int, default=[1.0], help="")
    parser.add_argument("--paddle_margin_list", nargs="+", type=int, default=[3], help="")
    parser.add_argument("--max_paddle_retrial", type=int, default=2, help="")

    args = parser.parse_args()

    # DS: snake length
    for cl in args.path_lengths:
        args.contour_length = cl
        args.distractor_length = cl / 3
        args.num_distractor_snakes = 30 / args.distractor_length
        dataset_subpath = 'curv_contour_length_' + str(cl)
        args.contour_path = os.path.join(args.output_dir, dataset_subpath)
        from_wrapper(args)
    args.contour_length = 6
    args.distractor_length = 2

    elapsed = time.time() - t
    print('n_totl_imgs (per condition) : ', str(args.num_samples))
    print('ELAPSED TIME OVER ALL CONDITIONS : ', str(elapsed))


if __name__ == '__main__':
    main()
