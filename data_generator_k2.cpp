#include <bits/stdc++.h>
using namespace std;
double M_PI = acos(-1);
int main(int argc, char** argv) {
	
	freopen("data_k2.data","w",stdout);
	int wide = 4;
	for(int i=0;i<2;i++)
	{
		int t = (i+1)*50;
		while(t--)
		{
			int c = (10*i+1)*wide;
			int l = c + rand()%(4*wide)-wide*2;
			double alpha = double(rand()%360)/360 * 2*M_PI;

			double x = cos(alpha) * l ,y = sin(alpha) * l ;
			cout<<x<<' '<<y<<' '<<i<<endl;
		} 
	}		
    return 0;
}