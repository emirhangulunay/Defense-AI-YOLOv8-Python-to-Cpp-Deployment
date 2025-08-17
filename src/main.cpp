#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>

using namespace cv;
using namespace cv::dnn;
using namespace std;

struct Detection {
	int classId;
	float confidence;
	Rect box;
};

static inline vector<string> loadNames(const string& path) {
	vector<string> names;
	ifstream ifs(path);
	if (!ifs.is_open()) return names;
	string line;
	while (getline(ifs, line)) {
		if (!line.empty() && line.back() == '\r') line.pop_back();
		if (!line.empty()) names.push_back(line);
	}
	return names;
}

static inline Rect clampRect(const Rect& r, const Size& sz) {
	Rect img(0, 0, sz.width, sz.height);
	Rect clipped = r & img;
	return clipped.area() > 0 ? clipped : Rect();
}

static vector<Detection> getDetections(const Mat& output, const Size& orig, float confThreshold, float nmsThreshold, int inputW, int inputH, bool isYOLOv8Layout)
{
	vector<Detection> dets;
	if (output.empty()) return dets;

	Mat out2d;
	if (output.dims == 3) {
		int d0 = output.size[0];
		int d1 = output.size[1];
		int d2 = output.size[2];
		if (d0 != 1) return dets;

		if (d1 > d2) {
			out2d = output.reshape(1, d1);
		}
		else {
			Mat tmp = output.reshape(1, d1);
			out2d = tmp.t();
		}
	}
	else if (output.dims == 2) {
		out2d = output;
	}
	else {
		return dets;
	}

	if (out2d.type() != CV_32F) {
		out2d = out2d.clone();
		out2d.convertTo(out2d, CV_32F);
	}

	if (out2d.cols < 6) {
		return dets;
	}

	static bool printedInfo = false;
	if (!printedInfo) {
		cout << "Output dims: " << output.dims << " sizes=[";
		for (int i = 0; i < output.dims; ++i) cout << output.size[i] << (i + 1 < output.dims ? "," : "");
		cout << "], out2d=" << out2d.rows << "x" << out2d.cols
			<< ", layout=" << (isYOLOv8Layout ? "YOLOv8 (no obj)" : "YOLOv5 (obj @4)") << endl;
		printedInfo = true;
	}

	vector<Rect> boxes;
	vector<float> scores;
	vector<int> classIds;

	const float sx = static_cast<float>(orig.width) / static_cast<float>(inputW);
	const float sy = static_cast<float>(orig.height) / static_cast<float>(inputH);

	for (int i = 0; i < out2d.rows; ++i) {
		const float* row = out2d.ptr<float>(i);
		const int clsStart = isYOLOv8Layout ? 4 : 5;
		const float obj = isYOLOv8Layout ? 1.0f : row[4];

		if (!isYOLOv8Layout && obj <= 1e-6f) continue;

		Mat scoresMat(1, out2d.cols - clsStart, CV_32F, (void*)(row + clsStart));
		double maxClsScore;
		Point maxLoc;
		minMaxLoc(scoresMat, nullptr, &maxClsScore, nullptr, &maxLoc);

		float conf = obj * static_cast<float>(maxClsScore);
		if (conf < confThreshold) continue;

		float x = row[0], y = row[1], w = row[2], h = row[3];
		bool normalized = (x <= 1.5f && y <= 1.5f && w <= 1.5f && h <= 1.5f);

		float cx_px, cy_px, w_px, h_px;
		if (normalized) {
			cx_px = x * orig.width;
			cy_px = y * orig.height;
			w_px = w * orig.width;
			h_px = h * orig.height;
		}
		else {
			cx_px = x * sx;
			cy_px = y * sy;
			w_px = w * sx;
			h_px = h * sy;
		}

		int left = static_cast<int>(cx_px - w_px / 2.0f);
		int top = static_cast<int>(cy_px - h_px / 2.0f);
		int ww = static_cast<int>(w_px);
		int hh = static_cast<int>(h_px);

		Rect box = clampRect(Rect(left, top, ww, hh), orig);
		if (box.area() <= 0) continue;

		boxes.push_back(box);
		scores.push_back(conf);
		classIds.push_back(maxLoc.x);
	}

	vector<int> keep;
	NMSBoxes(boxes, scores, confThreshold, nmsThreshold, keep);

	for (int idx : keep) {
		dets.push_back(Detection{ classIds[idx], scores[idx], boxes[idx] });
	}
	return dets;
}

int main() {
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

	const string modelPath = "models/best.onnx";
	const string videoPath = "videos/test.mp4";
	const string namesPath = "models/classes.txt";

	Net net = readNetFromONNX(modelPath);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	vector<string> classNames = loadNames(namesPath);

	VideoCapture cap(videoPath);
	if (!cap.isOpened()) {
		cerr << "Video açılamadı!" << endl;
		return -1;
	}

	const int inputW = 640, inputH = 640;
	float confThreshold = 0.25f;
	float nmsThreshold = 0.45f;

	Mat frame;
	int frameIdx = 0;

	// Pencereyi oluştur ve boyutlandır
	namedWindow("YOLO ONNX Video", WINDOW_NORMAL);
	resizeWindow("YOLO ONNX Video", 800, 600);

	while (cap.read(frame)) {
		try {
			Mat blob = blobFromImage(frame, 1.0f / 255.0f, Size(inputW, inputH),
				Scalar(), true, false);
			net.setInput(blob);

			Mat output = net.forward();
			if (output.empty()) {
				cerr << "Uyarı: Ağ çıktısı boş." << endl;
				continue;
			}

			bool isYOLOv8Layout = (output.dims == 3 && output.size[1] < output.size[2]);

			vector<Detection> dets = getDetections(output, frame.size(), confThreshold, nmsThreshold, inputW, inputH, isYOLOv8Layout);

			cout << "Frame " << frameIdx++ << " - Tespit (çizilecek): " << dets.size() << endl;

			for (const auto& d : dets) {
				rectangle(frame, d.box, Scalar(0, 255, 0), 2);
				string label;
				if (!classNames.empty() && d.classId >= 0 && d.classId < (int)classNames.size()) {
					label = classNames[d.classId];
				}
				else {
					label = string("id=") + to_string(d.classId);
				}
				label += " " + to_string(static_cast<int>(d.confidence * 100)) + "%";
				putText(frame, label, Point(d.box.x, max(0, d.box.y - 5)),
					FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
			}

			imshow("YOLO ONNX Video", frame);
			if (waitKey(1) == 27) break; 

		}
		catch (const cv::Exception& e) {
			cerr << "OpenCV Hatası: " << e.what() << endl;
		}
	}

	cap.release();
	destroyAllWindows();
	return 0;
}
