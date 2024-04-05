#include <opencv2/opencv.hpp>
#ifdef _WIN32
#include <windows.h>
#else
#include <X11/Xlib.h>
#endif

int getScreenWidth() {
    #ifdef _WIN32
    return GetSystemMetrics(SM_CXSCREEN);
    #else
    Display* disp = XOpenDisplay(NULL);
    Screen* scrn = DefaultScreenOfDisplay(disp);
    int width = scrn->width;
    XCloseDisplay(disp);
    return width;
    #endif
}

int getScreenHeight() {
    #ifdef _WIN32
    return GetSystemMetrics(SM_CYSCREEN);
    #else
    Display* disp = XOpenDisplay(NULL);
    Screen* scrn = DefaultScreenOfDisplay(disp);
    int height = scrn->height;
    XCloseDisplay(disp);
    return height;
    #endif
}

int main() {
    // Récupération des dimensions de l'écran
    int screenWidth = getScreenWidth();
    int screenHeight = getScreenHeight();

    // Chargement de l'image
    cv::Mat image = cv::imread("CarteVideDessus.jpg");
    if (image.empty()) {
        std::cerr << "Erreur lors du chargement de l'image." << std::endl;
        return -1;
    }

    // Redimensionnement de l'image selon les dimensions de l'écran
    cv::resize(image, image, cv::Size(screenWidth, screenHeight), 0, 0, cv::INTER_LINEAR);

    // Affichage de l'image
    char key = 0;
    while (key != 27) { // 27 est le code ASCII de la touche ESC
        cv::imshow("Image", image);
        key = cv::waitKey(1);
    }

    return 0;
}
