#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <vector>

int main() {
    // Paramètres de test
    const bool TEST = true;
    const float taille_reelle_aruco_mm = 42.0f; // Taille réelle de l'ArUco en millimètres

    cv::VideoCapture cap;
    if (!TEST) {
        cap.open(0); // Utiliser la caméra
        if (!cap.isOpened()) {
            std::cerr << "Erreur lors de l'ouverture de la caméra." << std::endl;
            return -1;
        }
    }

    // Création du détecteur ArUco
    cv::Ptr<cv::aruco::Dictionary> aruco_dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    cv::Ptr<cv::aruco::Detector> detector = cv::aruco::Detector::create(aruco_dict, parameters);

    std::vector<std::string> nomImages = {
        "CarteVideCote.jpg", "CarteVideDessus.jpg", "ConfigBlocsAvecArucoCote1.jpg",
        "ConfigBlocsAvecArucoCote2.jpg", "ConfigBlocsAvecArucoCote3.jpg", "ConfigBlocsAvecArucoDessusLoin1.jpg",
        "ConfigBlocsAvecArucoDessusLoin2.jpg", "ConfigBlocsAvecArucoDessusProche1.jpg",
        "ConfigBlocsAvecArucoDessusProche2.jpg", "ConfigBlocsSansAruco1.jpg", "ConfigBlocsSansAruco2.jpg"
    };
    int nbImages = nomImages.size();
    int cpt = 0;

    cv::Mat frame;
    if (TEST) {
        frame = cv::imread(nomImages[cpt]);
    }

    while (true) {
        if (!TEST) {
            cap >> frame;
        } else {
            frame = cv::imread(nomImages[cpt]);
            cv::resize(frame, frame, cv::Size(), 0.7, 0.7);
            if (cv::waitKey(1) == 'p') {
                cpt = (cpt + 1) % nbImages;
            }
        }

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        detector->detectMarkers(gray, corners, ids);

        if (!ids.empty()) {
            aruco::drawDetectedMarkers(frame, corners, ids);

            // Logique spécifique pour le traitement des marqueurs détectés ici...
            // Comme la détection des distances entre les marqueurs spécifiques et le dessin de textes.

        }

        cv::imshow("Aruco Detection", frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    if (!TEST) {
        cap.release();
    }
    cv::destroyAllWindows();
    return 0;
}
