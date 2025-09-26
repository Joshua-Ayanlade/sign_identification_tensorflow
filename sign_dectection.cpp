#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <raspicam/raspicam_cv.h>
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace raspicam;

int class_id;
float display_confidence;
Scalar color;

void Setup(int argc, char **argv, RaspiCam_Cv &Camera) {
    Camera.set(CAP_PROP_FRAME_WIDTH,  ("-w", argc, argv, 400));
    Camera.set(CAP_PROP_FRAME_HEIGHT, ("-h", argc, argv, 240));
    Camera.set(CAP_PROP_BRIGHTNESS,   ("-br", argc, argv, 50));
    Camera.set(CAP_PROP_CONTRAST,     ("-co", argc, argv, 50));
    Camera.set(CAP_PROP_SATURATION,   ("-sa", argc, argv, 50));
    Camera.set(CAP_PROP_GAIN,         ("-g", argc, argv, 50));
    Camera.set(CAP_PROP_FPS,          ("-fps", argc, argv, 100));
}

int main(int argc, char **argv) {
    string model_path = "Sign_detection_model/";
    vector<string> class_names = {"Not Detected", "Detected"};
    
    // Load the TensorFlow SavedModel
    Net net = readNet("symbol_detector.pb", "", "TensorFlow");
    
    if (net.empty()) {
        cerr << "Failed to load symbol detection model!" << endl;
        return -1;
    }
    
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    
    // Initialize Raspberry Pi Camera
    RaspiCam_Cv Camera;
    Setup(argc, argv, Camera);
    
    cout << "Connecting to camera" << endl;
    if (!Camera.open()) {
        cout << "Failed to Connect" << endl;
        return -1;
    }
    
    cout << "Camera Id = " << Camera.getId() << endl;
    cout << "Sign Detection Model Loaded Successfully!" << endl;
    cout << "Press 'q' to quit, 's' to save image, 'i' for info" << endl;
    
    // Model parameters
    int input_width = 100;
    int input_height = 100;
    double scale_factor = 1.0 / 255.0;
    Scalar mean_values = Scalar(0, 0, 0);
    bool swapRB = true;
    
    
    while(true) {
        auto start_total = chrono::system_clock::now();
        Mat frame;
        
        // Capture frame
        Camera.grab();
        Camera.retrieve(frame);
        
        if (frame.empty()) {
            cerr << "Failed to capture frame!" << endl;
            break;
        }
        
        // Sign detection inference
        auto start_inference = chrono::system_clock::now();
        
        Mat blob = blobFromImage(frame, scale_factor, Size(input_width, input_height), 
                                mean_values, swapRB, false);
        net.setInput(blob);
        Mat output = net.forward();
        
        auto end_inference = chrono::system_clock::now();
        
        // Process results
        float confidence = output.at<float>(0, 0);
        
        if (confidence > 0.7){
            class_id = 1;
        }
        else{
            class_id = 0;
            }
            
        if (class_id == 1){
            display_confidence = confidence;
        }
        else{
            display_confidence = 1 - confidence;
        }
        
        
        // Calculate timings
        auto end_total = chrono::system_clock::now();
        chrono::duration<double> inference_time = end_inference - start_inference;
        chrono::duration<double> total_time = end_total - start_total;
        
        double fps_total = 1.0 / total_time.count();
        double fps_inference = 1.0 / inference_time.count();
        
        // Display information
        string label = format("%s: %.1f%%", class_names[class_id].c_str(), confidence * 100);
        string inference_label = format("Inference: %.1fms", inference_time.count() * 1000);
        string fps_label = format("FPS: %.1f", fps_total);
        string confidence_label = format("Confidence: %.3f", confidence);
        
        if (class_id == 1){
            color = Scalar(0, 255, 0);
        }
        else{
            color = Scalar(0, 0, 255);
        }
        
        
        // Main result
        putText(frame, "SIGN DETECTION", Point(10, 25), 
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
        putText(frame, label, Point(10, 55), 
                FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
        
        // Performance info
        putText(frame, inference_label, Point(10, 80), 
                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(200, 200, 200), 1);
        putText(frame, fps_label, Point(150, 80), 
                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(200, 200, 200), 1);
        
        
        // Show frame
        namedWindow("Sign Detection",WINDOW_KEEPRATIO);
		resizeWindow("Sign Detection",640,320);
        imshow("Sign Detection", frame);
        
        // Console output (less frequent to avoid clutter)
        static int frame_count = 0;
        if (frame_count % 30 == 0) {
            cout << "FPS: " << fps_total << " | Inference: " << inference_time.count() * 1000 
                 << "ms | " << label << endl;
        }
        frame_count++;
        
        // Handle key presses
        int key = waitKey(1);
        if (key == 'q' || key == 'Q') {
            break;
        } else if (key == 's' || key == 'S') {
            string filename = format("Sign_detection_%d.jpg", (int)time(NULL));
            imwrite(filename, frame);
            cout << "Saved: " << filename << endl;
        } 
    }
    
    Camera.release();
    destroyAllWindows();
    cout << "Application terminated successfully!" << endl;
    
    return 0;
}
