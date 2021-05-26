#include "brisque.h";
#include <iostream>
#include <fstream>

//rescaling based on training data i libsvm
float rescale_vector[36][2];

using namespace std;

int read_range_file()
{
    //check if file exists
    char buff[100];
    int i;
    string range_fname = "allrange";
    FILE* range_file = fopen(range_fname.c_str(), "r");
    if (range_file == NULL) return 1;
    //assume standard file format for this program	
    fgets(buff, 100, range_file);
    fgets(buff, 100, range_file);
    //now we can fill the array
    for (i = 0; i < 36; ++i)
    {
        float a, b, c;
        fscanf(range_file, "%f %f %f", &a, &b, &c);
        rescale_vector[i][0] = b;
        rescale_vector[i][1] = c;
    }
    return 0;
}


int main(int argc, char** argv)
{
    //trainModel(188);
    //float qualityscore;
    //qualityscore = computescore("i.jpg");
    //cout << "Quality Score: " << qualityscore << endl;
    //ofstream result("result.txt");
    //result << qualityscore;
    //result.close();
    
    if (argc < 2) 
    {
        cerr << "Input Image argument not given." << endl;
        return -1;
    }

    //read in the allrange file to setup internal scaling array
    if (read_range_file()) {
        cerr << "Unable to open allrange file." << endl;
        return -1;
    }

    if (argv[1] == string("train")) 
    { 
        if(argc == 3) trainModel(stoi(argv[2]));
        else 
        {
            cerr << "Number of training images argument not given." << endl;
            return -1;
        }
    } 
    else
    {
        float qualityscore;
        qualityscore = computescore(argv[1]);
        cout << "Quality Score: " << qualityscore << endl;
        ofstream result("result.txt");
        result << qualityscore;
        result.close();
    }
}

