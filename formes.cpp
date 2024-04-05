#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    cv::VideoCapture cap(0); // Ouvrir la caméra par défaut
    if (!cap.isOpened()) {
        std::cout << "Erreur lors de l'ouverture de la caméra." << std::endl;
        return -1;
    }

    cv::Mat frame, gray, blurred, edges;
    std::vector<std::vector<cv::Point>> contours;

    while (true) {
        cap >> frame; // Lire une nouvelle frame
        if (frame.empty()) {
            std::cout << "Erreur lors de la capture de l'image." << std::endl;
            break;
        }

        // Convertir en niveaux de gris
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Appliquer un flou
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

        // Détection de contours
        cv::Canny(blurred, edges, 50, 150, 3);

        // Trouver les contours
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (size_t i = 0; i < contours.size(); i++) {
            // Approximation des contours à des polygones + vérification du nombre de côtés
            std::vector<cv::Point> approx;
            cv::approxPolyDP(contours[i], approx, 0.03 * cv::arcLength(contours[i], true), true);
            std::string shape = "Polygone";

            switch (approx.size()) {
                case 3: shape = "Triangle"; break;
                case 4: shape = "Rectangle"; break; // Peut être un carré ou un rectangle
                case 5: shape = "Pentagone"; break;
                case 6: shape = "Hexagone"; break;
                case 7: shape = "Heptagone"; break;
                case 8: shape = "Octogone"; break;
                default: break;
            }

            // Dessiner le polygone et afficher le type
            cv::drawContours(frame, std::vector<std::vector<cv::Point>>{approx}, -1, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, shape, approx[0], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
        }

        // Afficher le résultat
        cv::imshow("Polygones détectés", frame);

        // Quitter avec la touche 'q'
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release(); // Libérer la caméra
    cv::destroyAllWindows(); // Fermer toutes les fenêtres

    return 0;
}
