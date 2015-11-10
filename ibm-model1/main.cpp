//
//  main.cpp
//  ibm-model1
//
//  Template code by mtemms
//  Created by David Kelly on 31/10/2015.
//  Copyright © 2015 David Kelly. All rights reserved.
//

#include<vector>
#include<string>
#include<iostream>
#include<iomanip>
#include<cmath>

using namespace std;

// want to represents vocab items by integers because then various tables
// need by the IBM model and EM training can just be represented as 2-dim
// tables indexed by integers

// the following #defines, defs of VS, VO, S, O, and create_vocab_and_data()
// are set up to deal with the specific case of the two pair corpus
// (la maison/the house)
// (la fleur/the flower)

// S VOCAB
#define LA 0
#define MAISON 1
#define FLEUR 2
// O VOCAB
#define THE 0
#define HOUSE 1
#define FLOWER 2

#define VS_SIZE 3
#define VO_SIZE 3
#define D_SIZE 2

#define EM_ITERATIONS 3
#define PRINT_EXPECTATION_TABLE

vector<string> VS(VS_SIZE); // S vocab: VS[x] gives Src word coded by x
vector<string> VO(VO_SIZE); // O vocab: VO[x] gives Obs word coded by x

vector<vector<int> > S(D_SIZE); // all S sequences; in this case 2
vector<vector<int> > O(D_SIZE); // all O sequences; in this case 2

vector<vector<float>> tr_table; // tr(o|s)

// sets S[0] and S[1] to be the int vecs representing the S sequences
// sets O[0] and O[1] to be the int vecs representing the O sequences
void create_vocab_and_data();

// initialises the tr_table with uniform probabilities
void init_translation_probability();

// prints the table of translation probabilities
void print_translation_probability();

// the actual logic of the ibm model 1 algorithm
void ibm_model1_em();

// prints the pairs of words and their probability
void print_pairs_tr_prob();

// functions which use VS and VO to 'decode' the int vecs representing the
// Src and Obs sequences
void show_pair(int d);
void show_O(int d);
void show_S(int d);

int main() {
    create_vocab_and_data();
    
    // guts of it to go here
    // you may well though want to set up further global data structures
    // and functions which access them
    
    cout << endl << "Initialised tr(o|s) with uniform values: " << endl;
    init_translation_probability();
    print_pairs_tr_prob();
    cout << endl;
    
    
    for(int i = 0; i < EM_ITERATIONS; i++){
        cout << "EM Iteration #" << i << ": " << endl;
        ibm_model1_em();
        cout << "tr(o|s):" << endl;
        print_pairs_tr_prob();
        cout << endl;
    }
    
    return 0;
}

void print_pairs_tr_prob(){
    for(int i = 0; i < tr_table.size(); i++){
        for(int j = 0; j < tr_table[i].size(); j++){
            cout << VO[j] << " \t\t " << VS[i] << "\t\t\t" << tr_table[j][i] << endl;
        }
    }
}

void ibm_model1_em(){
    // initialise counts to 0
    vector<vector<float>> count;
    count.resize(VO_SIZE);
    for(int i = 0; i < count.size(); i++){
        count[i].resize(VS_SIZE);
        for(int j = 0; j < count[i].size(); j++){
            count[i][j] = 0;
        }
    }
    
    // E
    for(int k = 0; k < D_SIZE; k++){ // for each sentence pair in corpus
        for(int j = 0; j < O[k].size(); j++){ // for each o[j] in sentence
            int o_word = O[k][j];
            // calculate denominator
            float denom = 0.0;
            for(int idenom = 0; idenom < S[k].size(); idenom++){
                denom += tr_table[o_word][S[k][idenom]];
            }
            for(int i = 0; i < S[k].size(); i++){ // for each s[i] in sentence
                int s_word = S[k][i];
                // calculate numerator
                float numer = tr_table[o_word][s_word];
                count[o_word][s_word] += numer / denom; // count[o][s] += P((j, i)|o, s)
            }
        }
    }
    
#ifdef PRINT_EXPECTATION_TABLE
    cout << "Unnormalised counts E(o|s): " << endl;
    for(int i = 0; i < count.size(); i++){
        for(int j = 0; j < count[i].size(); j++){
            cout << VO[j] << " \t\t" << VS[i] << "\t\t\t" << count[j][i] << endl;
        }
    }
    cout << endl;
#endif
    
    // M
    for(int i = 0; i < VS_SIZE; i++){ // for each s in Vs
        // calculate denominator
        float denom = 0.0;
        for(int idenom = 0; idenom < VO_SIZE; idenom++){ // Sum for each #(o[idenom], s[i])
            denom += count[idenom][i];
        }
        for(int j = 0; j < VO_SIZE; j++){
            float numer = count[j][i];
            tr_table[j][i] = numer / denom;
        }
    }
}

void print_translation_probability(){
    // print head row
    cout << "tr(o|s)";
    for(int i = 0; i < VS_SIZE; i++){
        cout << " || " << i;
    }
    cout << endl << "------------------------" << endl;
    // print each row
    for(int i = 0; i < VS_SIZE; i++){
        cout << "   " << i << "   ";
        for(int j = 0; j < VO_SIZE; j++){
            cout << " || " << tr_table[i][j];
        }
        cout << endl;
    }
}

void init_translation_probability(){
    int total_size = VS_SIZE * VO_SIZE;
    // create rows
    tr_table.resize(VO_SIZE);
    for(int i = 0; i < tr_table.size(); i++){
        // create cols
        tr_table[i].resize(VS_SIZE);
        for(int j = 0; j < tr_table[i].size(); j++){
            tr_table[i][j] = (1 / (float)total_size); // initialise with uniform probabilities
        }
    }
}

void create_vocab_and_data() {
    
    VS[LA] = "la";
    VS[MAISON] = "maison";
    VS[FLEUR] = "fleur";
    
    VO[THE] = "the";
    VO[HOUSE] = "house";
    VO[FLOWER] = "flower";
    
    cout << "source vocab\n";
    for(int vi=0; vi < VS.size(); vi++) {
        cout << VS[vi] << " ";
    }
    cout << endl;
    cout << "observed vocab\n";
    for(int vj=0; vj < VO.size(); vj++) {
        cout << VO[vj] << " ";
    }
    cout << endl;
    
    S[0].resize(2);   O[0].resize(2);
    // make S[0] be {LA,MAISON}
    //      O[0] be {THE,HOUSE}
    S[0][0] = LA; S[0][1] = MAISON;
    O[0][0] = THE; O[0][1] = HOUSE;
    
    // make S[1] be {LA,FLEUR}
    //      O[1] be {THE,FLOWER}
    S[1].resize(2);   O[1].resize(2);
    S[1][0] = LA; S[1][1] = FLEUR;
    O[1][0] = THE; O[1][1] = FLOWER;
    
    for(int d = 0; d < S.size(); d++) {
        show_pair(d);
    }
}

void show_O(int d) {
    for(int i=0; i < O[d].size(); i++) {
        cout << VO[O[d][i]] << " ";
    }
}

void show_S(int d) {
    for(int i=0; i < S[d].size(); i++) {
        cout << VS[S[d][i]] << " ";
    }
}

void show_pair(int d) {
    cout << "S" << d << ": ";
    show_S(d);
    cout << endl;
    cout << "O" << d << ": ";
    show_O(d);
    cout << endl;
}
