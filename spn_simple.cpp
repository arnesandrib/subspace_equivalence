#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstring>
#include <iomanip>

using namespace std;


/*
G0-G15 representatives from Leander-Poschmann
*/

const vector<vector<uint8_t>> sboxer = {
    {0, 1, 2, 13, 4, 7, 15, 6, 8, 11, 12, 9, 3, 14, 10, 5},
    {0, 1, 2, 13, 4, 7, 15, 6, 8, 11, 14, 3, 5, 9, 10, 12},
    {0, 1, 2, 13, 4, 7, 15, 6, 8, 11, 14, 3, 10, 12, 5, 9},
    {0, 1, 2, 13, 4, 7, 15, 6, 8, 12, 5, 3, 10, 14, 11, 9},
    {0, 1, 2, 13, 4, 7, 15, 6, 8, 12, 9, 11, 10, 14, 5, 3},
    {0, 1, 2, 13, 4, 7, 15, 6, 8, 12, 11, 9, 10, 14, 3, 5},
    {0, 1, 2, 13, 4, 7, 15, 6, 8, 12, 11, 9, 10, 14, 5, 3},
    {0, 1, 2, 13, 4, 7, 15, 6, 8, 12, 14, 11, 10, 9, 3, 5},
    {0, 1, 2, 13, 4, 7, 15, 6, 8, 14, 9, 5, 10, 11, 3, 12},
    {0, 1, 2, 13, 4, 7, 15, 6, 8, 14, 11, 3, 5, 9, 10, 12},
    {0, 1, 2, 13, 4, 7, 15, 6, 8, 14, 11, 5, 10, 9, 3, 12},
    {0, 1, 2, 13, 4, 7, 15, 6, 8, 14, 11, 10, 5, 9, 12, 3},
    {0, 1, 2, 13, 4, 7, 15, 6, 8, 14, 11, 10, 9, 3, 12, 5},
    {0, 1, 2, 13, 4, 7, 15, 6, 8, 14, 12, 9, 5, 11, 10, 3},
    {0, 1, 2, 13, 4, 7, 15, 6, 8, 14, 12, 11, 3, 9, 5, 10},
    {0, 1, 2, 13, 4, 7, 15, 6, 8, 14, 12, 11, 9, 3, 10, 5}
};


const uint8_t M16[16][16] = {
    {0,1,1,1,0,1,1,0,1,1,0,0,0,1,1,0},
    {0,0,1,1,1,0,1,1,0,1,1,0,0,0,1,1},
    {1,0,0,1,1,1,0,1,1,0,1,1,0,0,0,1},
    {1,1,0,0,1,1,1,0,1,1,0,1,1,0,0,0},
    {0,1,1,0,0,1,1,1,0,1,1,0,1,1,0,0},
    {0,0,1,1,0,0,1,1,1,0,1,1,0,1,1,0},
    {0,0,0,1,1,0,0,1,1,1,0,1,1,0,1,1},
    {1,0,0,0,1,1,0,0,1,1,1,0,1,1,0,1},
    {1,1,0,0,0,1,1,0,0,1,1,1,0,1,1,0},
    {0,1,1,0,0,0,1,1,0,0,1,1,1,0,1,1},
    {1,0,1,1,0,0,0,1,1,0,0,1,1,1,0,1},
    {1,1,0,1,1,0,0,0,1,1,0,0,1,1,1,0},
    {0,1,1,0,1,1,0,0,0,1,1,0,0,1,1,1},
    {1,0,1,1,0,1,1,0,0,0,1,1,0,0,1,1},
    {1,1,0,1,1,0,1,1,0,0,0,1,1,0,0,1},
    {1,1,1,0,1,1,0,1,1,0,0,0,1,1,0,0}
};

vector<uint8_t> SBOX(16);
random_device rd;
mt19937 gen(rd());

int product(uint8_t x, uint8_t y) {
    return __builtin_popcount(x & y) & 1;
}

uint16_t v2i(const vector<int>& v) {
    uint16_t res = 0;
    for (size_t i = 0; i < v.size(); i++) {
        res |= (v[i] << i);
    }
    return res;
}



std::vector<uint8_t> evaluate_matrix(const std::vector<uint8_t>& v) {

    std::vector<uint8_t> result(16, 0);

    for (int i = 0; i < 16; i++) { 
        uint8_t sum = 0;
        for (int j = 0; j < 16; j++) {
            if (M16[i][j] == 1){
                sum ^= (v[j]);
            }
        }
        result[i] = sum;
    }
    
    return result;
}


void encrypt(vector<uint8_t>& state, const vector<vector<uint8_t>>& keys, int nr) {

    for (int round = 0; round < nr; round++) {
        for (int i = 0; i < 16; i++) {
         state[i] ^= keys[round][i];
        }
    
        for (int i = 0; i < 16; i++) {
            state[i] = SBOX[state[i]];
        }
       state = evaluate_matrix(state);
        
        
    }

}

uint32_t run_experiment(int m, int nr, uint8_t U1, uint8_t U2, const vector<uint8_t>& S,uint32_t* H) {
    vector<vector<uint8_t>> keys(nr + 1, vector<uint8_t>(16));
    uniform_int_distribution<> dis(0, 15);
    uniform_int_distribution<> dis_S(0, S.size() - 1);

    for (int i = 0; i <= nr; i++) {
        for (int j = 0; j < 16; j++) {
            keys[i][j] = dis(gen);
        }
    }
    uint32_t unique = 0;

    
    for (int iter = 0; iter < m; iter++) {
        vector<uint8_t> a(16), b(16);
        vector<uint8_t> diff(16);
        
        // Generate random a
        for (int i = 0; i < 16; i++) {
            a[i] = dis(gen);
        }
        
        // Generate random diff from S
        for (int i = 0; i < 1; i++) {
            diff[i] = S[dis_S(gen)];
        }
        
        // Create b with difference in first position
        b = a;
        for(int i=0; i < 1; i++){
            b[i] ^=   diff[i];
        }

        
        vector<uint8_t> ca = a, cb = b;
        encrypt(ca, keys, nr);
        encrypt(cb, keys, nr);

        vector<uint8_t> diff_out(16);
        for (int i = 0; i < 16; i++) {
            diff_out[i] = ca[i] ^ cb[i];
        }
        vector<int> s_bits(16);

        for (int i = 0; i < 16; i++) {
            s_bits[i] = product(diff_out[i], U2);
        }
        uint16_t s = v2i(s_bits);

        if (H[s] == 0)
            unique++;
        H[s]++;
     
    }
    return unique;
}

int main() {
    const int m_iterations = 65536;
    const int N = m_iterations;
    const int K = 65536;
    
    for (int t = 0; t < 16; t++) {
        SBOX = sboxer[t];
        cout << "###############"<<endl;
        cout << "S-Box G"<<t<<" "<<endl;
        uint32_t maxx = 0;
        int maxwt = 100;
        uint32_t minuq = 100000000;
         /*
         Expected number of bins with a ball when throwing
         2^16 balls into 2^16 bins
         */
        float exp = 41427;
        /*
         Standard deviation of expected unique elements when throwing
         2^16 balls into 2^16 bins
         */
        float stdev = 80; 
        // hyperplanes of GF(2)^4 U = {x | ux=0}
        for (uint8_t u1 = 1; u1 < 16; u1++) {

            uint8_t u2 = u1;
            int rounds = 3;
            
            vector<uint8_t> S;
            for (int x = 1; x < 16; x++) {
                if (product(u1, x) == 0) {
                    S.push_back(x);
                }
            }
            
            uint32_t H[65536] = {0};
            uint32_t unique; 
            unique=run_experiment(m_iterations, rounds, u1, u2, S,H);

            if (unique < minuq){
                minuq = unique;
            }
            std::vector<std::pair<uint32_t, uint16_t>> top_list;
            for (int i = 0; i < 65536; i++) {
                if (H[i] > 0) {
                    top_list.push_back({H[i], (uint16_t)i});
                }
            }

            std::sort(top_list.rbegin(), top_list.rend());
            if (top_list[0].first > maxx){
                maxx = top_list[0].first;
            }
                       
    }
    /*
    minuq: The minimal size of unique cosets hit over the 15 non-zero masks u1 (each define a hyperlane)
    stdev: Number of standard deviations from expected # of non-empty bins
    maxx: The maximum value 
    */
    cout << "minuq " << " #stdev" <<  "    maxco" <<endl;
    cout << minuq<<"  "<<(exp-minuq)/stdev<<"    "<<maxx<<"   "<<endl;
            cout << "###############"<<endl;
            cout<<endl;
    }
    
    return 0;
}
