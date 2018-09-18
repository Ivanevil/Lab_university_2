#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
using namespace std;
int main()
{
	ifstream myFile;
	myFile.open("matrixes.txt", ios_base::in);
	if (myFile.is_open())
	{
		cout << "OK"<<endl;
		string file;
		//while (myFile.good())
		//{
			getline(myFile, file);
			cout << file << endl;
			istringstream iss(file);
			vector <string> mas;
			copy(istream_iterator<std::string>(iss),
			istream_iterator<std::string>(),
			back_inserter<std::vector<std::string> >(mas));
			cout << mas.size() << endl;
			stringstream str1;
			str1 << mas[0];
			int m = 0;
			str1 >> m;
			cout << m <<endl;
			stringstream str2;
			str2 << mas[1];
			int n = 0;
			str2 >> n;
			cout << n <<endl;
			int **matrix = new int* [m]; // две строки в массиве
			for (int count = 0; count < 2; count++)
				matrix[count] = new int [n]; 
			for (int i=0 ;i<m; i++)
			{
				getline(myFile, file);
				cout << file << endl;
				istringstream iss(file);
				vector <string> mas;
				copy(istream_iterator<std::string>(iss),
				istream_iterator<std::string>(),
				back_inserter<std::vector<std::string> >(mas));
				cout << mas.size() << endl;

			}
		//}
		
		myFile.close();
	}
	else 
	{
		cout <<"fail"<<endl;
	}
	system ("pause");
	return 0;
}