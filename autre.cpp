#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>

int main() {
    // Charger l'image contenant les marqueurs ArUco
    cv::Mat arucoImage = cv::imread("ArucoBoard.png");
    cv::imshow("Image", arucoImage);
    cv::waitKey(0);

    // Préparation du dictionnaire et des paramètres de détection ArUco
    cv::Ptr<cv::aruco::Dictionary> arucoDict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_100);
    cv::Ptr<cv::aruco::DetectorParameters> arucoParams = cv::aruco::DetectorParameters::create();

    // Détecter les marqueurs ArUco dans l'image
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    cv::aruco::detectMarkers(arucoImage, arucoDict, markerCorners, markerIds, arucoParams, rejectedCandidates);

    if (!markerCorners.empty()) {
        markerIds = markerIds; // Assurez-vous que markerIds est un vecteur plat

        for (size_t i = 0; i < markerCorners.size(); i++) {
            // Convertir les coins pour faciliter l'accès
            std::vector<cv::Point2f> corners = markerCorners[i];
            cv::Point2f topRight = corners[1];
            cv::Point2f bottomRight = corners[2];
            cv::Point2f bottomLeft = corners[3];
            cv::Point2f topLeft = corners[0];

            // Dessiner les lignes entre les coins
            cv::line(arucoImage, topLeft, topRight, cv::Scalar(255, 0, 0), 2);
            cv::line(arucoImage, topRight, bottomRight, cv::Scalar(255, 0, 0), 2);
            cv::line(arucoImage, bottomRight, bottomLeft, cv::Scalar(255, 0, 0), 2);
            cv::line(arucoImage, bottomLeft, topLeft, cv::Scalar(255, 0, 0), 2);

            // Calculer et dessiner le point central
            cv::Point center = (topLeft + bottomRight) / 2.0;
            cv::circle(arucoImage, center, 4, cv::Scalar(0, 255, 0), -1);

            // Afficher l'ID du marqueur ArUco sur l'image
            std::string markerIdStr = std::to_string(markerIds[i]);
            cv::putText(arucoImage, markerIdStr, topLeft, cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 2);

            std::cout << "Aruco Marker ID: " << markerIds[i] << std::endl;
            cv::imshow("Image", arucoImage);
            cv::waitKey(0);
        }
    }

    cv::destroyAllWindows();
    return 0;
}
