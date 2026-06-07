# plot.py

## 处理pred结果的.json文件,画图
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random


def plot_img_and_mask(img, mask, index, epoch, save_dir):
    # ... (此函数无需修改)
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    # plt.show()
    plt.savefig(save_dir + "/batch_{}_{}_seg.png".format(epoch, index))


def show_seg_result(img, result, index, epoch, save_dir=None, is_ll=False, palette=None, is_demo=False, is_gt=False):
    # img = mmcv.imread(img)
    # img = img.copy()
    # seg = result[0]
    if palette is None:
        palette = np.random.randint(
            0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3  # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2

    if not is_demo:
        color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[result == label, :] = color
    else:
        # ==================== 代码修改开始 ====================
        # result 是一个元组: (da_seg_mask, ll_seg_mask)
        # da_seg_mask 包含类别: 0 (背景), 1 (直接行驶区域), 2 (间接行驶区域)
        # ll_seg_mask 包含类别: 0 (背景), 1 (实线), 2 (虚线)

        color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)

        # 1. 为可行驶区域分配颜色
        #    您可以根据需要自定义这些颜色 (BGR格式)
        da_mask = result[0]
        color_area[da_mask == 2] = [0, 255, 0]  # 绿色: 间接行驶区域
        color_area[da_mask == 1] = [255, 200, 0]  # 黄色: 直接行驶区域

        # 2. 为车道线分配颜色
        #    车道线会覆盖在可行驶区域之上，这是正确的行为
        ll_mask = result[1]
        color_area[ll_mask == 1] = [255, 0, 0]  # 实线
        color_area[ll_mask == 2] = [0, 255, 255]  #  虚线

        color_seg = color_area
        # ==================== 代码修改结束 ====================

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    # print(color_seg.shape)
    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    # img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)

    if not is_demo:
        if not is_gt:
            if not is_ll:
                cv2.imwrite(save_dir + "/batch_{}_{}_da_segresult.png".format(epoch, index), img)
            else:
                cv2.imwrite(save_dir + "/batch_{}_{}_ll_segresult.png".format(epoch, index), img)
        else:
            if not is_ll:
                cv2.imwrite(save_dir + "/batch_{}_{}_da_seg_gt.png".format(epoch, index), img)
            else:
                cv2.imwrite(save_dir + "/batch_{}_{}_ll_seg_gt.png".format(epoch, index), img)
    return img


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # ... (此函数无需修改)
    tl = line_thickness or round(0.0001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # if label:
    #     tf = max(tl - 1, 1)  # font thickness
    #     t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    #     c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    #     cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    #     print(label)
    # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


if __name__ == "__main__":
    pass
# ... (文件其余部分无需修改)