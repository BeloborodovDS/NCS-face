#ifndef DETECTION_LAYER_H
#define DETECTION_LAYER_H

#include <opencv2/opencv.hpp>
#include <vector>

/*
 Get detection boxes and probabilities of detections from YOLO output.
 YOLO output size is [side * side * [5*num + classes]]
 For each cell of [side*side] grid it contains:
 - Four coordinates for each of [num] boxes
 - One score for each of [num] boxes
 - One score for each of [classes] classes
 
 Params:
 predictions     : pointer to output from YOLO
 w,h             : image size
 thresh          : probabilities are clipped to 0 if <thresh
 probs, boxes    : Bounding boxes and corresponding probabilities (possibly 0) - OUTPUT
 only_objectness : if true, return only probability of (any) object
 side            : side of blocks grid (from network params)
 num             : number of boxes pes block (from network params)
 classes         : number of classes (from network params)
 sqrt            : whether box sizes are squared or not (from network params)
 */
void get_detection_boxes(float* predictions, int w, int h, float thresh, 
			 std::vector<float>& probs, std::vector<cv::Rect>& boxes, int only_objectness=0,
			 int side=11, int num=2, int classes=1, int sqrt=1
			);

/*
 * intersection/union
 */
float box_iou(cv::Rect a, cv::Rect b);

/*
 * non-maximim suppression
 * @param boxes: bounding boxes
 * @param probs: probabilities
 * @param classes: number of classes
 * @param thresh:  thresh for iou to merge boxes
 */
void do_nms(std::vector<cv::Rect>& boxes, std::vector<float>& probs, int classes, float thresh);

#endif
