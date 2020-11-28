#%% [markdown]
# 获取元数据

#%%
import os
import numpy as np 
import pandas as pd 
import cv2 as cv 

# %%
# img_roi 必须是深拷贝，否则影响之后的均值计算
def label_region(img_roi, file_name, start_w = 612, start_h = 162, w = 10, h = 16):
    results_path = "dataset/results"
    img_roi[start_w: start_w + w, start_h: start_h + h] = [0, 0, 255]
    cv.imwrite(os.path.join(results_path, file_name), img_roi)

# %%
# img 必须是单通道的灰度图
def calculate_mean(img, start_w = 612, start_h = 162, w = 10, h = 16):
    s = 0
    for i in range(start_w, start_w + w):
        for j in range(start_h, start_h + h):
            # print(img_gray_roi[i, j])
            s += img[i, j]

    avg = int(s / (w * h))

    return avg

#%%
def main():
    root_path = "dataset/images" 
    results = {"file_name":[], "avg":[]}

    start_w = 612
    start_h = 162
    w = 10
    h = 16

    for i, file_name in enumerate(os.listdir(root_path)):
        # print(item)
        # print(root_path, file_name)
        img = cv.imread(os.path.join(root_path, file_name))
        assert img.shape == (1016, 960, 3)

        img_roi = img.copy()

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_gray_roi = img_gray.copy()

        # img_roi 必须是深拷贝，否则影响之后的均值计算
        label_region(img_roi, file_name, start_w, start_h, w, h)
        
        # img 必须是单通道的灰度图
        avg = calculate_mean(img_gray_roi, start_w, start_h, w, h)

        results["file_name"].append(file_name)
        results["avg"].append(avg)

        # print("No.", i, "'s mean is ", avg)


        if i % 50 == 0:
            print("Processing No.", int(i))

    df = pd.DataFrame(results)
    df.to_csv("results.csv", index_label="Number")
    print("Successfully done!")
    pass

# %%
if __name__ == "__main__":
    main()
    pass


