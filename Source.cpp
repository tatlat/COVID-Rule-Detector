// program3.cpp
// This program detects whether people are following the rules of covid-19:
// wearing face masks, staying 6 feet apart, and not touching their faces.
// It classifies people as wearing masks, not wearing masks, or improper 
// face mask usage (not covering the nose). Proper social distancing is 
// indicated by green lines between people while improper distancing is 
// indicated by red lines.
// The skin segmentation and back projection code is based off of
// https://docs.opencv.org/3.4/da/d7f/tutorial_back_projection.html
// estimating depth is based on the pinhole projection model
// https://en.wikipedia.org/wiki/Pinhole_camera_model
// estimating focal length 
// https://www.edmundoptics.com/knowledge-center/application-notes/imaging/understanding-focal-length-and-field-of-view/
// Author: Tanvir Tatla

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <cmath>
#include <sstream>

constexpr auto minDistance = 1828; // millimeters, equivalent to 6 feet
using namespace cv;

// Pose - position
// x1 is the leftmost x coordinate, x2 is rightmost
// z is depth and sometimes used as y2
struct Pose {
  int x1 = 0;
  int x2 = 0;
  int y = 0;
  float z = 0;
};

// loadClassifiers - loads the xml files for the Haar Cascade Classifiers
// preconditions: the files for each classifier is in the same directory as
//                this file
// postconditions: classifers are loaded
void loadClassifiers(CascadeClassifier& faceCascade, 
  CascadeClassifier& noseCascade, CascadeClassifier& mouthCascade) {

  faceCascade.load("haarcascade_frontalface_alt2.xml");
  noseCascade.load("Nose.xml");
  mouthCascade.load("Mouth.xml");
}

// noFace - Writes "no face detected" at top of frame
// preconditions: N/A
// postconditions: adds text to the frame
void noFace(Mat& frame) {
  Point p(frame.cols / 4, 30);
  Scalar color(255, 255, 255);
  putText(frame, "No face detected", p, FONT_HERSHEY_SIMPLEX, 1.0, color);
}
 
// analyzeFace - looks at features of the face in faceBox and returns an 
//        integer based on whether it finds a mouth or nose
// preconditions: N/A
// postconditions: returns an int corresponding to the face mask truth table
//                 1 means no mouth, no nose, so yes mask
//                 0 means no mouth or nose, so no mask
//                -1 means no mouth, yes nose, so improper mask
int analyzeFace(const Mat& image, const Rect& faceBox, 
  CascadeClassifier& noseCascade, CascadeClassifier& mouthCascade) {

  Mat face = image(faceBox); // face subimage
  std::vector<Rect> noses, mouths;

  // detect noses and mouth in face subimage, store in vectors
  noseCascade.detectMultiScale(face, noses, 1.1, 2, 0 |
    CASCADE_SCALE_IMAGE, Size(25, 15));
  mouthCascade.detectMultiScale(face, mouths, 1.1, 2, 0 |
    CASCADE_SCALE_IMAGE, Size(25, 15));

  // no mouth, no nose, so likely wearing mask
  if (mouths.empty() && noses.empty()) {
    return 1;
  }

  // mouth and nose? likely no mask
  else if (mouths.size() && noses.size()) {
    return 0;
  }

  // improper mask
  return -1;
}


// classify - labels the face and encloses it in a rectangle based on the
//      mask classification from analyzeFace
// preconditions: none
// postconditions: draws a red rectangle when no mask, yellow for improper,
//      green for mask
void classify(Mat& frame, Rect& face, int mask) {
  Scalar color;
  String message;

  switch (mask) {
  case 0: // no mask
    color = Scalar(0, 0, 255); // red
    message = "NO MASK";
    break;

  case 1: // wearing mask
    color = Scalar(0, 255, 0); // green
    message = "HAS MASK";
    break;

  default: // mask doesn't cover nose, improper
    color = Scalar(0, 255, 255); // yellow
    message = "IMPROPER MASK";
    break;
  }

  // draw rectangle and text
  rectangle(frame, face, color, 3);
  Point p(face.x, face.y - 5);
  putText(frame, message, p, FONT_HERSHEY_SIMPLEX, 1.0, color);
}

// detectMask - detects and classifies each face based on their mask usage
// preconditions - none
// postconditions - labels each face in the image by drawing a rectangle
//          and writing text over the face locations
void detectMask(Mat& image, Mat& dst, std::vector<Rect> faces,
  CascadeClassifier& noseCascade, CascadeClassifier& mouthCascade) {

  // loop through every face
  for (int i = 0; i < faces.size(); i++) {
    // analyze and classify face
    int mask = analyzeFace(image, faces[i], noseCascade, mouthCascade);
    classify(dst, faces[i], mask);
  }
}

// detectFace - detects every face in frame
// preconditions - none
// postconditions - returns the number of faces found
//        stores every detected face in the faces vector
int detectFace(Mat& image, CascadeClassifier& faceCascade, 
  std::vector<Rect>& faces) {

  // detect and store every face in input image
  faceCascade.detectMultiScale(image, faces, 1.1, 2, 0 |
    CASCADE_SCALE_IMAGE, Size(30, 30));

  return faces.size();
}

// estimateDepth - uses the eye and it's average width of 24mm to estimate
//          distance of a person from the webcam
// preconditions: none
// postconditions: returns a float representing an estimation of how far away
//           a person is from the camera
// see https://en.wikipedia.org/wiki/Pinhole_camera_model
// and https://www.edmundoptics.com/knowledge-center/application-notes/imaging/understanding-focal-length-and-field-of-view/
float estimateDepth(const Rect& face, double& ppm, double& focal, 
  const int& fov) {
  double headWidth = 165; // average width is 24mm
  double facePixels = face.width;

  // if pixels per millimeter or focal length not calculated
  if (ppm == 0) {
    const int w = 320, h = 240;
    double d = hypot(w, h); // fov is diagonal
    double theta = (fov * (3.14 / 180)) / 2; // convert fov to radians
    focal = (d / 2) * (cos(theta) / sin(theta)); // focal length
    ppm = facePixels / headWidth;
  }

  return float((headWidth * focal) / facePixels); // distance from camera
}


// estimatePostion - estimates the position of a person in 3d space
// preconditions - none
// postconditions - returns a Pose struct with the x and y coordianates in
//          pixels. The z value is given in millimeters.
Pose estimatePosition(const Rect& faceBox, double& ppm, double& focal,
  const int& fov) {

  Pose p;
  p.z = estimateDepth(faceBox, ppm, focal, fov);
  p.x1 = faceBox.x;
  p.x2 = faceBox.x + faceBox.width;
  p.y = faceBox.y;

  return p;
}

// calculateDistance - calculates the real world 3d distance between 2 poses
//           in millimeters
// preconditions - none
// postconditions - returns the distance between 2 poses using distance formula
double calculateDistance(const Pose& p1, const Pose& p2, const double& ppm) {
  Pose left = (p1.x1 < p2.x1) ? p1 : p2; // get leftmost point
  Pose right = (p1.x1 < p2.x1) ? p2 : p1;

  if (ppm == 0) return 0;

  double dx = (right.x1 - left.x2) / ppm; // divide by ppm to get millimeters
  double dy = (right.y - left.y) / ppm;
  double dz = (right.z - left.z);

  // distance = sqrt(dx^2 + dy^2 + dz^2)

  // square deltas
  dx *= dx;
  dy *= dy;
  dz *= dz;

  double distance = dx + dy + dz; // add deltas
  distance = sqrt(distance); // square root to get distance

  return distance;
}

// checkDistance - checks if the distance is less than 6 feet and draws
//       a red line between 2 poses if it is and a green line if more than 
//       6 feet
// preconditions - none
// postconditions - draws a line on the frame between the poses
void checkDistance(Mat& frame, int distance, Pose p1, Pose p2) {
  Scalar color;

  // not distancing
  if (distance < minDistance) {
    color = Scalar(0, 0, 255); // red
  }

  else { // proper distancing
    color = Scalar(0, 255, 0); // green
  }

  // draw lines from middle of top of head
  Point first((p1.x1 + p1.x2) / 2, p1.y);
  Point second((p2.x1 + p2.x2) / 2, p2.y);

  if (first.x > 0 && second.x > 0) {
    line(frame, first, second, color, 2);
  }
}

// socialDistance - tells you who is social distancing in frame
// preconditions - none
// postconditions - draws a red or green line betwen every person in frame
//                  green = proper distance, red = less than 6 feet
void socialDistance(const Mat& in, Mat& dst, std::vector<Rect> faces, 
  double& ppm, double& focal, const int& fov) {

  std::vector<Pose> positions;

  // for each face
  for (int i = 0; i < faces.size(); i++) {
    Mat f = in(faces[i]); // roi is face in input image
      // estimate the real world postion of that face in a 3d face
      // and store in vector
    positions.push_back(
      estimatePosition(faces[i], ppm, focal, fov)
    );
  }

  // if only one person/pose, then stop
  if (positions.size() <= 1) return;

  // for each person
  for (int i = 0; i < positions.size() - 1; i++) {
    for (int j = i + 1; j < positions.size(); j++) {
      // calculate the distance between every person
      int dist = calculateDistance(positions[i], positions[j], ppm);
      checkDistance(dst, dist, positions[i], positions[j]);
    }
  }
}

// coverFace - covers the face and neck in in image based on the coordinates
//        of our face Rect
// preconditions - none
// postconditions - draws a solid black box over the in image   
void coverFace(const Mat& in, Pose& fPose, const Rect& face) {
  fPose.x1 = face.x;
  fPose.x2 = face.x + face.width;
  fPose.y = face.y;
  fPose.z = face.y + face.height;

  // draw a solid rectangle at face location
  Rect face2 = face;
  face2.height *= 1.5;
  rectangle(in, face2, Scalar(0, 0, 0), -1);
}

// isTouchingFace - tells you when someone is touching their face with their hand
// preconditions - hitBox is the rectangle with location of the hand. Offset
//             should not be too big, less than 5 and greater than -1 is ideal.
// postconditions - returns true when the hitBox rectangle overlaps with face
//            rectangle.
bool isTouchingFace(const Rect& face, const Rect& hitBox, const int& offset) {
  // get top left and bottom right points of both rectangles
  Point faceTL = face.tl(), faceBR = face.br();
  Point hitTL = hitBox.tl(), hitBR = hitBox.br();

  faceTL.x += offset; faceBR.x -= offset; hitTL.x += offset; hitBR.x -= offset;
  faceTL.y += offset; faceBR.y -= offset; hitTL.y += offset; hitBR.y -= offset;

  // if one rectangle is to the right of the other, then not touching
  if (faceTL.x >= hitBR.x || hitTL.x >= faceBR.x) return false;

  // if one rectangle is below the other, then not touching
  if (faceTL.y >= hitBR.y || hitTL.y >= faceBR.y) return false;

  return true; // otherwise they must be touching
}

// findBiggestContour - finds the biggest contour that is smaller than our
//       threshold area and sets the bound around it
// preconditions - none
// postconditions - returns the index of the largest contour in the contour
//           vector, sets the bounding box around the largest contour
int findBiggestContour(const std::vector<std::vector<Point>>& contours,
  Rect& bound) {

  // threshold area, hand won't be that big so if a contour bigger than
  // the threshold, then it's probably not a hand.
  int threshold = 26880;
  int index = -1;
  int size = 0;

  // for each contour
  for (int i = 0; i < contours.size(); i++) {
    // check it's size and area
    if (contours[i].size() > size && contourArea(contours[i]) < threshold) {
      size = int(contours[i].size());
      index = i;
    }
  }

  // if we found a contour that's under threshold and large
  // set the bound around it
  if (index != -1) {
    std::vector<Point> poly;
    approxPolyDP(Mat(contours[index]), poly, 3, true);
    bound = boundingRect(Mat(poly));
  }

  return index;
}

// backProject - calculates the backprojection of frame from hsv's histogram
// preconditions - hsv is a hue saturation value image
// postconditions - returns an image representing the backprojection of 
//            frame using hsv as a histogram
Mat backProject(const Mat& hsv, const Mat& frame) {
  // set up histogram
  int hbins = 12, sbins = 12; // number of bins in histogram
  MatND hist;
  int histSize[] = { hbins, sbins };
  float hue_range[] = { 0, 180 };
  float sat_range[] = { 0, 256 };
  const float* ranges[] = { hue_range, sat_range };
  int channels[] = { 0, 1 };

  /// Get the Histogram and normalize it
  calcHist(&hsv, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
  normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());

  // convert to hsv so it matches histogram
  Mat hsvFrame = frame.clone();
  GaussianBlur(frame, hsvFrame, Size(7, 7), 20, 60);
  cvtColor(hsvFrame, hsvFrame, COLOR_BGR2HSV);
  //dilate(hsvFrame, hsvFrame, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)));
  //erode(hsvFrame, hsvFrame, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)));

  /// Get Backprojection
  Mat backproj;
  calcBackProject(&hsvFrame, 1, channels, hist, backproj, ranges, 1, true);

  // remove noise
  dilate(backproj, backproj, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)));
  dilate(backproj, backproj, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)));
  //GaussianBlur(backproj, backproj, Size(7, 7), 20, 60);

  /*imshow("bp", backproj);
  if (waitKey(0) != 'n') {
    imwrite("bp.jpg", backproj);
  }*/

  return backproj;
}

// segmentSkinColor - segments the skin color in frame by using the face image
//            to construct a hue saturation histogram and backprojection 
//            to find skin colored pixels in the frame
// preconditions - none
// postconditions - returns a Pose struct with the x and y coordianates in
//          pixels. The z value is given in millimeters.
// Based off of https://docs.opencv.org/3.4/da/d7f/tutorial_back_projection.html
void segmentSkinColor(const Mat& face, const Mat& frame, Rect& touchBox) {
  Mat hsv;
  cvtColor(face, hsv, COLOR_BGR2HSV); // hsv better with poor lighting

  Mat bp = backProject(hsv, frame); // get back projection

  std::vector<std::vector<Point>> contours;
  std::vector<Vec4i> hierarchy;
  int thresh = 200;

  // threshold values that are unlikely to be skin 
  threshold(bp, bp, thresh, 255, THRESH_BINARY);
  /*imshow("thresh", bp);
  if (waitKey(0) != 'n') {
    imwrite("thresh.jpg", bp);
  }*/

  // find contours in our thresholded image
  findContours(bp, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE,
    Point(0, 0));

  // find the biggest contour and set our touchBox bound around it
  int c = findBiggestContour(contours, touchBox);

  //Mat contourFrame = Mat::zeros(bp.size(), CV_8UC1);
  //drawContours(contourFrame, contours, c, Scalar(255), 3, 8, hierarchy);

  /*rectangle(frame, touchBox.tl(), touchBox.br(), Scalar(0, 0, 255), 2);
  imshow("contour", frame);
  if (waitKey(0) != 'n') {
    std::stringstream ss;
    ss << "rectangle" << counter << ".jpg";
    imwrite(ss.str(), frame);
    counter++;
  }*/
}

// checkFaceTouch - checks whether the person in frame is touching their face.
// preconditions - none
// postconditions - returns true if someone is touching their face, false if
//            not. Writes text on dst image if they are touching their face
bool checkFaceTouch(const Mat& in, Mat& dst, const Rect& faceBox) {
  Mat face = in(faceBox);
  Mat covered = dst.clone();
  Pose fPose;
  Rect touchBox;

  // cover the face so it isn't detected as a hand
  coverFace(covered, fPose, faceBox);

  // separate skin colored pixels from non-skin colored
  segmentSkinColor(face, covered, touchBox); 

  int offset = 4;

  // check if hand is touching face
  bool touch = isTouchingFace(faceBox, touchBox, offset); 

  // draw text if touching
  if (touch) {
    Point p(faceBox.x, faceBox.y + (faceBox.height / 2));
    Scalar color(255, 0, 0);
    putText(dst, "No touch face!", p, FONT_HERSHEY_SIMPLEX, 1.0, color, 2);
  }

  return touch;
}

// main - turns on the webcam and sets the capture to 320x240p.
//		Continuously searches for faces, analyzes facial features to 
//		determine mask usage, estimates 3d real world position to check
//		for proper social distancing, and checks for hand on face touching
//    until the user exits by hitting 'q'. Can only detect face touching
//    when one person is on screen. Can only detect masks when not 
//    touching the face
// precondition: If one additional argument is added, the program assumes
//    it can be converted to an integer.
//    The haar cascade xml files for face, mouth, nose, and eyes need to
//	  be in the same directory as this program. They should be titled
//    haarcascade_frontalface_alt2.xml, haarcascade_eye.xml Nose.xml, and 
//    Mouth.xml.
// postconditions: Opens images that indicate whether people in the webcam
//		frame are (not) wearing face masks, (not) properly social distancing,  
//		and touching their face with their hand. Exits when the user hits 'q'
int main(int argc, char** argv) {
  int fov = 80;

  // if a second argument was included, then set the diagonal field of view
  if (argc == 2) {
    fov = atoi(argv[1]);
  }

  CascadeClassifier faceCascade;
  CascadeClassifier noseCascade;
  CascadeClassifier mouthCascade;

  // load the pretrained models
  loadClassifiers(faceCascade, noseCascade, mouthCascade);

  VideoCapture cap(0);

  // if the files didn't load or webcam didn't turn on, then stop
  if (faceCascade.empty() || noseCascade.empty() || mouthCascade.empty()
    || !cap.isOpened()) {
    return 1;
  }

  // Set video to 320x240
  cap.set(CAP_PROP_FRAME_WIDTH, 320);
  cap.set(CAP_PROP_FRAME_HEIGHT, 240);

  Mat frame;
  int numFaces = 0;
  double focalLength = 0;
  double ppm = 0; // pixels per millimeter

  // while user doesn't enter q (for quit)
  while (waitKey(15) != 'q') {
    cap >> frame; // get frame from webcam

    // couldn't get frame so stop
    if (frame.empty())
      break;

    // flip the image so it isn't mirrored
    flip(frame, frame, 1);

    // set to grey so feature detection/matching can work
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    bool touchingFace = false;
    std::vector<Rect> faces;
    numFaces = detectFace(gray, faceCascade, faces); // detect faces in frame

    // write no face if nothing detected
    if (numFaces == 0) {
      noFace(frame);
    }

    // check for face touching if only one person in frame
    else if (numFaces == 1) {
      touchingFace = checkFaceTouch(frame, frame, faces[0]);
    }

    // check social distancing if multiple ppl in frame
    else if (numFaces > 1) {
      socialDistance(gray, frame, faces, ppm, focalLength, fov);
    }

    // numFaces can't be negative one, but if it is...
    else {
      std::cout << "error" << std::endl;
    }

    // only check for face masks when not touching to prevent 
    // false results
    if (!touchingFace) {
      detectMask(gray, frame, faces, noseCascade, mouthCascade);
    }

    // show the results
    imshow("video", frame);
  }

  // release the webcam and stop recording video
  cap.release();

  // close all windows
  destroyAllWindows();
  return 0;
}