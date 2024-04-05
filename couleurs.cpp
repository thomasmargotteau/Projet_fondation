#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main() {
    cv::VideoCapture cap(0); // Utilisation de la première caméra

    if (!cap.isOpened()) {
        std::cerr << "Erreur : la caméra ne peut pas être ouverte." << std::endl;
        return -1;
    }

    cv::Mat frame, hsvFrame;
    cv::namedWindow("frame", cv::WINDOW_AUTOSIZE);

    // Définition des plages de couleurs HSV pour le filtre
    cv::Scalar blueMin(94, 80, 2), blueMax(120, 255, 255);
    cv::Scalar redMin(136, 87, 111), redMax(180, 255, 255);
    cv::Scalar greenMin(25, 52, 72), greenMax(102, 255, 255);
    cv::Scalar whiteMin(0, 0, 200), whiteMax(180, 30, 255);
    cv::Scalar blackMin(0, 0, 0), blackMax(180, 255, 30);

    std::vector<std::pair<cv::Scalar, cv::Scalar>> colorRanges = {
        {blueMin, blueMax}, {redMin, redMax}, {greenMin, greenMax},
        {whiteMin, whiteMax}, {blackMin, blackMax}
    };

    std::vector<std::string> colorNames = {"Bleu", "Rouge", "Vert", "Blanc", "Noir"};
    std::vector<cv::Scalar> drawColors = {
        cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0),
        cv::Scalar(255, 255, 255), cv::Scalar(0, 0, 0)
    };

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    while (true) {
        bool ret = cap.read(frame);
        if (!ret) {
            std::cerr << "Erreur : impossible d'accéder aux données de la caméra." << std::endl;
            break;
        }

        cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);

        for (size_t i = 0; i < colorRanges.size(); ++i) {
            cv::Mat mask;
            cv::inRange(hsvFrame, colorRanges[i].first, colorRanges[i].second, mask);
            mask = cv::dilate(mask, kernel);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

            for (size_t j = 0; j < contours.size(); ++j) {
                double area = cv::contourArea(contours[j]);
                if (area > 300) {
                    cv::Rect boundingBox = cv::boundingRect(contours[j]);
                    cv::rectangle(frame, boundingBox, drawColors[i], 2);
                    cv::putText(frame, colorNames[i], cv::Point(boundingBox.x, boundingBox.y),
                                cv::FONT_HERSHEY_SIMPLEX, 1.0, drawColors[i]);
                }
            }
        }

        cv::imshow("frame", frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
