#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <vector>
#include <iomanip>

using namespace cv;
using namespace std;

int main() {
    string imagePath = "resources/test_image_2.jpg";
    Mat originalImage = imread(imagePath, IMREAD_COLOR);

    
    // 步骤1: 应用高斯模糊
    Mat blurredImage;
    GaussianBlur(originalImage, blurredImage, Size(11, 11), 0);

    // 步骤2: 转换为灰度图
    Mat grayImage;
    cvtColor(blurredImage, grayImage, COLOR_BGR2GRAY);
    
    // 步骤3: 图像二值化
    Mat binaryImage;
    threshold(grayImage, binaryImage, 230, 255, THRESH_BINARY);
    
    // 步骤4: 形态学开运算
    Mat morphImage;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
    morphologyEx(binaryImage, morphImage, MORPH_OPEN, kernel);

    // 步骤5: Canny边缘检测
    Mat cannyImage;
    Canny(morphImage, cannyImage, 50, 150);
    
    // 步骤6: 识别细长轮廓，画出最小外接矩形
    Mat resultImage = originalImage.clone();
    
    // 寻找轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(morphImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 为每个轮廓绘制最小外接矩形
    for (size_t i = 0; i < contours.size(); i++) {
        // 过滤掉太小的轮廓
        double area = contourArea(contours[i]);
        if (area < 500) { // 最小面积阈值
            continue;
        }
        
        // 计算最小外接矩形
        RotatedRect minRect = minAreaRect(contours[i]);
        
        float width = minRect.size.width;
        float height = minRect.size.height;
        float aspectRatio = max(width, height) / min(width, height); 
        
        if (aspectRatio < 2.0) {
            continue;
        }
        
        // 获取矩形的四个顶点
        Point2f vertices[4];
        minRect.points(vertices);
        
        // 绘制最小外接矩形
        for (int j = 0; j < 4; j++) {
            line(resultImage, vertices[j], vertices[(j + 1) % 4], Scalar(0, 255, 0), 2);
        }      

    }
    
    // 收集所有细长轮廓的最小外接矩形
    vector<RotatedRect> elongatedRects;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area < 500) continue;
        RotatedRect minRect = minAreaRect(contours[i]);
        float width = minRect.size.width;
        float height = minRect.size.height;
        float aspectRatio = max(width, height) / min(width, height);
        if (aspectRatio < 2.0) continue;
        elongatedRects.push_back(minRect);
    }

    // 配对并框选
    for (size_t i = 0; i < elongatedRects.size(); i++) {
        for (size_t j = i + 1; j < elongatedRects.size(); j++) {
            // 判断长边方向是否平行（角度差小于10度）
            float angle1 = elongatedRects[i].angle;
            float angle2 = elongatedRects[j].angle;
            float angleDiff = fabs(angle1 - angle2);
            if (angleDiff > 5.0) continue;
            // 判断长边长度是否相近（相差不超过20%）
            float len1 = max(elongatedRects[i].size.width, elongatedRects[i].size.height);
            float len2 = max(elongatedRects[j].size.width, elongatedRects[j].size.height);
            float lenRatio = len1 / len2;
            if (lenRatio < 0.9 || lenRatio > 1.1) continue;
            // 获取两个矩形的所有顶点
            vector<Point2f> allVertices;
            Point2f v1[4], v2[4];
            elongatedRects[i].points(v1);
            elongatedRects[j].points(v2);
            for (int k = 0; k < 4; k++) allVertices.push_back(v1[k]);
            for (int k = 0; k < 4; k++) allVertices.push_back(v2[k]);
            // 用boundingRect框住所有顶点
            Rect bigRect = boundingRect(allVertices);
            rectangle(resultImage, bigRect, Scalar(0, 0, 255), 3);
        }
    }

    // 保存处理步骤的中间结果
    imwrite("results/01_blurred.jpg", blurredImage);
    imwrite("results/02_gray.jpg", grayImage);
    imwrite("results/03_binary.jpg", binaryImage);
    imwrite("results/04_morph.jpg", morphImage);
    imwrite("results/05_canny.jpg", cannyImage);
    imwrite("results/06_final_result.jpg", resultImage);

    return 0;
}
