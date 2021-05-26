#include "brisque.h"
#include <sstream>

void trainModel(int n)
{

    try 
    {
        cout << "retraining..." << endl;

        string foldername = "images";
        vector<float> scores;

        ifstream scorefile("scores.txt");
        float s;
        while (scorefile >> s) scores.push_back(s);
        scorefile.close();

        const char* filename = "train.txt";
        ofstream trainfile(filename);
        trainfile.close();

        for (int i = 0; i < n; i++)
        {
            string imname = foldername + "/" + to_string(i) + ".bmp";
            cout << imname << " " << scores[i] << endl;

            cv::Mat orig = cv::imread(imname.c_str());
            vector<double> brisqueFeatures;
            ComputeBrisqueFeature(orig, brisqueFeatures);
            //printVector(brisqueFeatures);
            printVectortoFile(filename, brisqueFeatures, scores[i]);

        }

        system("svm-scale.exe -l -1 -u 1 -s allrange train.txt > train_scale");
        system("svm-train.exe -s 3 -g 0.05 -c 1024 -b 1 -q train_scale allmodel");

        remove("train.txt");
        remove("train_scale");
    }
    catch (cv::Exception e) 
    {
        std::cout << e.msg;
    }
    
}

