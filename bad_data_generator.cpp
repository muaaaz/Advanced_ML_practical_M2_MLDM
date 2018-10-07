#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
	
	freopen("data1.data","w",stdout);
	int wide = 4;
	for(int i=0;i<5;i++)
	{
		int t = 20;
		while(t--)
		{
			int c = (2*i+1)*wide;
			int x = rand()%wide-wide/2,y = rand()%wide-wide/2;
			if(x > 0)
				x = x+c;
			else
				x = x-c;
			if(y > 0)
				y = y+c;
			else
				y = y-c;
			cout<<x<<' '<<y<<' '<<i<<endl;
		} 
	}		
    return 0;
}