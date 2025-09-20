#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    Mat original = imread("resources/test_image.png");
    
    // ===================== 一、图像颜色空间转换 =====================
    
    // 1. 转换为灰度图并保存
    Mat gray;
    cvtColor(original, gray, COLOR_BGR2GRAY);
    imwrite("resources/1_1_gray.jpg", gray);
    
    // 2. 转换为HSV图片并保存
    Mat hsv;
    cvtColor(original, hsv, COLOR_BGR2HSV);
    imwrite("resources/1_2_hsv.jpg", hsv);
    
    // ===================== 二、应用滤波操作 =====================
    
    // 1. 应用均值滤波并保存
    Mat mean_filtered;
    blur(original, mean_filtered, Size(15, 15));
    imwrite("resources/2_1_mean_filter.jpg", mean_filtered);
    
    // 2. 应用高斯滤波并保存
    Mat gaussian_filtered;
    GaussianBlur(original, gaussian_filtered, Size(15, 15), 0);
    imwrite("resources/2_2_gaussian_filter.jpg", gaussian_filtered);
    
    // ===================== 三、特征提取 =====================
    
    // 1. 提取红色区域（用HSV方法）并保存
    Mat red_mask;
    // 定义红色HSV范围
    Scalar lower_red1 = Scalar(0, 50, 50);
    Scalar upper_red1 = Scalar(15, 255, 255);
    Scalar lower_red2 = Scalar(120, 50, 50);
    Scalar upper_red2 = Scalar(180, 255, 255);
    
    Mat mask1, mask2;
    inRange(hsv, lower_red1, upper_red1, mask1);
    inRange(hsv, lower_red2, upper_red2, mask2);
    red_mask = mask1 | mask2;
    
    Mat red_region;
    original.copyTo(red_region, red_mask);
    imwrite("resources/3_1_red_region.jpg", red_region);
    
    // 2. 寻找图像中红色的外轮廓并保存
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(red_mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    Mat contour_image = original.clone();
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(contour_image, contours, (int)i, Scalar(0, 255, 0), 2);
    }
    imwrite("resources/3_2_red_contours.jpg", contour_image);
    
    // 3. 寻找图像中红色的bounding box并保存
    Mat bbox_image = original.clone();
    for (size_t i = 0; i < contours.size(); i++) {
        Rect bbox = boundingRect(contours[i]);
        rectangle(bbox_image, bbox, Scalar(255, 0, 0), 2);
    }
    imwrite("resources/3_3_red_bounding_box.jpg", bbox_image);
    
    // 4. 计算轮廓的面积并保存（在图像上标注面积）
    Mat area_image = original.clone();
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > 100) { // 过滤小面积
            Moments M = moments(contours[i]);
            Point2f center(M.m10/M.m00, M.m01/M.m00);
            putText(area_image, "Area: " + to_string((int)area), 
                   Point(center.x-30, center.y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
        }
    }
    imwrite("resources/3_4_contour_areas.jpg", area_image);
    
    // 5. 提取高亮区域并进行图形学处理
    Mat bright_gray;
    cvtColor(original, bright_gray, COLOR_BGR2GRAY);
    
    // 二值化提取高亮区域
    Mat binary;
    threshold(bright_gray, binary, 100, 255, THRESH_BINARY);
    
    // 膨胀
    Mat dilated;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(binary, dilated, kernel);
    
    // 腐蚀
    Mat eroded;
    erode(dilated, eroded, kernel);
    
    // 漫水处理
    Mat floodfilled = eroded.clone();
    floodFill(floodfilled, Point(0, 0), Scalar(128));
    
    imwrite("resources/3_5_bright_processed.jpg", floodfilled);
    
    // ===================== 四、图像绘制 =====================
    
    // 1. 绘制任意圆形方形和文字
    Mat drawing = original.clone();
    circle(drawing, Point(100, 100), 50, Scalar(255, 0, 0), 3);
    rectangle(drawing, Point(200, 50), Point(300, 150), Scalar(0, 255, 0), 3);
    putText(drawing, "DXRMV", Point(50, 250), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    imwrite("resources/4_1_shapes_text.jpg", drawing);
    
    // 2. 绘制红色的外轮廓
    Mat red_contours_only = Mat::zeros(original.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(red_contours_only, contours, (int)i, Scalar(0, 255, 0), 2);
    }
    imwrite("resources/4_2_red_contours_only.jpg", red_contours_only);
    
    // 3. 绘制红色的bounding box
    Mat bbox_only = Mat::zeros(original.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++) {
        Rect bbox = boundingRect(contours[i]);
        rectangle(bbox_only, bbox, Scalar(255, 0, 0), 2);
    }
    imwrite("resources/4_3_red_bbox_only.jpg", bbox_only);
    
    // ===================== 五、对图像进行处理 =====================
    
    // 1. 图像旋转35度
    Point2f center(original.cols/2.0, original.rows/2.0);
    Mat rotation_matrix = getRotationMatrix2D(center, 35, 1.0);
    Mat rotated;
    warpAffine(original, rotated, rotation_matrix, original.size());
    imwrite("resources/5_1_rotated_35.jpg", rotated);
    
    // 2. 图像裁剪为左上角四分之一
    Rect crop_region(0, 0, original.cols/2, original.rows/2);
    Mat cropped = original(crop_region);
    imwrite("resources/5_2_cropped_quarter.jpg", cropped);
    
    return 0;
}
