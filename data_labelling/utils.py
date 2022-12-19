def plot_image(ax, img, title, fontsize=12):
    ax.imshow(img, cmap='gray')
    ax.locator_params(nbins=3)
    #     ax.set_xlabel('x-label', fontsize=fontsize)
    #     ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)


def create_yolo_training_sample(crops_dict, class_index):
    # this function creates training sample in yolo format
    ##    (0,0)--------(1,0)
    #       |parent image|
    #       |            |
    #     (0,1)---------(1,1)()
    #   class x_center y_center width height  note: its not comma seperated
    #     img_p = cv.imread(resized_parent_image_path)
    #     imp_c = cv.imread(cropped_image_path)
    #     resized_image_shape = img_p.shape

    #     parent_image_x = resized_image_shape[1]

    start_x = round(crops_dict['startX'])
    start_y = round(crops_dict['startY'])
    end_x = round(crops_dict['endX'])
    end_y = round(crops_dict['endY'])
    #     cropped_image_shape = img_c.shape
    x_center = round(0.5 * (start_x + end_x) / parent_image_x, 6)
    y_center = round(0.5 * (start_y + end_y) / parent_image_y, 6)
    width = round(crops_dict["width"] / parent_image_x, 6)
    height = round(crops_dict["height"] / parent_image_y, 6)
    return class_index, x_center, y_center, width, height
