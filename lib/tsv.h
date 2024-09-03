// Literally just a class for reading TSVs and turning them 
//		into a basic matrix/dataframe thing

#include <vector>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>

//////////////////////////////////////
template <typename type>
class TSV {

public:

	// File
	std::string filename;

	// Data
	std::vector<std::string> row_names;
	std::vector<std::string> col_names;
	std::vector<std::vector<type>> data;

    // Stats
	int cols;
    int rows;
    int size;
    size_t bytes;

	// Empty
	TSV() {};

	// Proper constructor
	TSV(std::string _filename) {
		filename = _filename;	
	}

	// Copy constructor
	TSV(const TSV &_TSV) {
		filename = _TSV.filename;
		row_names = _TSV.row_names;
		col_names = _TSV.col_names;
		data = _TSV.data;
	}

	void read_delim(const char &del) {

		std::fstream f(filename);

		// Make sure the file is open
   		if(!f.is_open()) throw std::runtime_error("[ ERROR: Could not open file! ]");

   		type val;
		bool row = true;
   		int col_index = 0;
		std::string line, col;

		// Read in column header
		std::getline(f, line);
        std::stringstream s1(line);
        while(std::getline(s1, col, del)){
            col_names.push_back(col); 
        }

        int num_cols = col_names.size() + 1;

        while (std::getline(f, line)) {

            std::stringstream s2(line);

	        while (std::getline(s2, col, del)) {

			  	if (col_index % num_cols == 0) {
					row_names.push_back(col);
					data.push_back({});
				} else {
					std::stringstream s2(col);
					if (s2 >> val) {
						data.at((col_index - 1) / num_cols).push_back(val);
					} else {
						throw std::runtime_error("[ ERROR: Invalid type in data! ]");
					}
				}
				col_index += 1;
		   	}
			row = true;
		}

		f.close();

		// Collect Stats
		cols = col_names.size();
	    rows = row_names.size();
	    size = cols * rows;
        bytes = size * sizeof(int);

	}

	// Flatten out table into matrix
	type* flatten(const char &major_order) {

	    type *mat = new type[size];

		if (major_order == 'C') {	
		    for (int i = 0; i < rows; i++) {
		        for (int j = 0; j < cols; j++) {
		            for (int k = 0; k < rows; k++) {
		                mat[j * rows + k] = data.at(k).at(j);
		            }  
		        }
		    }
		} else {
			for (int i = 0; i < rows; i++) {
		        for (int j = 0; j < cols; j++) {
		                mat[i * cols + j] = data.at(i).at(j);
		        }
		    }	
		}

	    return mat;
	}
}; 