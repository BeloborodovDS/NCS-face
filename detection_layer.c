#include "detection_layer.h"

#include <opencv2/opencv.hpp>
#include <vector>

void get_detection_boxes(float* predictions, int w, int h, float thresh, 
			 std::vector<float>& probs, std::vector<cv::Rect>& boxes, 
			 int only_objectness, int side, int num, int classes, int sqrt
			)
{
    int i,j,n;
    //for all blocks in grid
    for (i = 0; i < side*side; ++i)
    {
        int row = i / side;
        int col = i % side;
	//for all [num] boxes
        for(n = 0; n < num; ++n)
	{
            //calculate box parameters
	    int index = i*num + n;
            int p_index = side*side*classes + i*num + n;
            float scale = predictions[p_index];
            int box_index = side*side*(classes + num) + (i*num + n)*4;
	    cv::Rect box;
            box.x = (predictions[box_index + 0] + col) / side * w;
            box.y = (predictions[box_index + 1] + row) / side * h;
            box.width = pow(predictions[box_index + 2], (sqrt?2:1)) * w;
            box.height = pow(predictions[box_index + 3], (sqrt?2:1)) * h;
	    box.x -= 0.5*box.width;
	    box.y -= 0.5*box.height;
	    boxes.push_back(box);
	    //calculate probability
            if(only_objectness)
	    {
                //probability of (any) object
		//probs[index][0] = scale;
	        probs.push_back(scale);
            }
            else
	    {
		//per-class probability
		for(j = 0; j < classes; ++j)
		{
		    int class_index = i*classes;
		    float prob = scale*predictions[class_index+j];
		    //probs[index][j] = (prob > thresh) ? prob : 0;
		    probs.push_back( (prob > thresh) ? prob : 0);
		}
	    }
        }
    }
}