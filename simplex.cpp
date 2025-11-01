// simplex.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>
#include <cstring> // strdup
#include <limits>
#include "json.hpp"

using json = nlohmann::json;
using namespace std;

using Matrix = vector<vector<double>>;
using Vector = vector<double>;

double clean(double val, double eps = 1e-10) {
    return (fabs(val) < eps) ? 0.0 : val;
}

//  helpers
Matrix multiply(const Matrix& A, const Matrix& B) {
    if (A.empty() || B.empty() || A[0].size() != B.size())
        throw invalid_argument("Invalid matrix dimensions for multiplication");
    size_t n = A.size();
    size_t m = B[0].size();
    size_t p = B.size();
    Matrix C(n, Vector(m, 0.0));
    for (size_t i = 0; i < n; ++i)
        for (size_t k = 0; k < p; ++k)
            for (size_t j = 0; j < m; ++j)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

Vector multiply(const Matrix& A, const Vector& v) {
    if (A.empty() || A[0].size() != v.size())
        throw invalid_argument("Invalid dimensions for matrix-vector multiplication");
    Vector result(A.size(), 0.0);
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < v.size(); ++j)
            result[i] += A[i][j] * v[j];
    return result;
}

Vector multiply(const Vector& v, const Matrix& A) {
    if (A.empty() || v.size() != A.size())
        throw invalid_argument("Invalid dimensions for vector-matrix multiplication");
    size_t m = A[0].size();
    Vector result(m, 0.0);
    for (size_t j = 0; j < m; ++j)
        for (size_t i = 0; i < v.size(); ++i)
            result[j] += v[i] * A[i][j];
    return result;
}

Matrix transpose(const Matrix& A) {
    if (A.empty()) return Matrix{};
    size_t n = A.size();
    size_t m = A[0].size();
    Matrix T(m, Vector(n, 0.0));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j)
            T[j][i] = A[i][j];
    return T;
}

Matrix inverse(const Matrix& A) {
    size_t n = A.size();
    if (n == 0 || A[0].size() != n)
        throw invalid_argument("Matrix must be square for inversion");
    Matrix aug(n, Vector(2 * n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) aug[i][j] = A[i][j];
        aug[i][n + i] = 1.0;
    }
    for (size_t i = 0; i < n; ++i) {
        double maxEl = fabs(aug[i][i]);
        size_t maxRow = i;
        for (size_t k = i + 1; k < n; ++k) {
            if (fabs(aug[k][i]) > maxEl) {
                maxEl = fabs(aug[k][i]);
                maxRow = k;
            }
        }
        if (maxEl < 1e-12) throw runtime_error("Matrix is singular and cannot be inverted");
        swap(aug[i], aug[maxRow]);
        double diag = aug[i][i];
        for (size_t j = 0; j < 2 * n; ++j) aug[i][j] /= diag;
        for (size_t k = 0; k < n; ++k) {
            if (k != i) {
                double factor = aug[k][i];
                for (size_t j = 0; j < 2 * n; ++j) aug[k][j] -= factor * aug[i][j];
            }
        }
    }
    Matrix inv(n, Vector(n, 0.0));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) inv[i][j] = aug[i][j + n];
    return inv;
}

Matrix selectColumns(const Matrix& A, const vector<size_t>& colIndices) {
    if (A.empty()) return Matrix{};
    size_t n = A.size();
    size_t m = A[0].size();
    for (size_t idx : colIndices) if (idx >= m) throw invalid_argument("Column index out of range");
    Matrix result(n, Vector(colIndices.size(), 0.0));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < colIndices.size(); ++j)
            result[i][j] = A[i][colIndices[j]];
    return result;
}

Vector selectElements(const Vector& v, const vector<size_t>& indices) {
    Vector result; result.reserve(indices.size());
    for (size_t idx : indices) {
        if (idx >= v.size()) throw invalid_argument("Index out of range");
        result.push_back(v[idx]);
    }
    return result;
}

double dot(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) throw invalid_argument("Vectors must be of same size for dot product");
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) result += a[i] * b[i];
    return result;
}

int sgn(double val) {
    if (val > 0) return 1;
    if (val < 0) return -1;
    return 0;
}

Vector add(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) throw invalid_argument("Vectors must be of same size for addition");
    Vector result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] + b[i];
    return result;
}

Vector getColumn(const Matrix& A, size_t col) {
    Vector result(A.size());
    for (size_t i = 0; i < A.size(); ++i) result[i] = A[i][col];
    return result;
}

double distance_to_segment(double x, double d_niz, double d_ver) {
    if (x < d_niz) return d_niz - x;
    if (x > d_ver) return x - d_ver;
    return 0.0; 
}
// runSimplex

string runSimplex(const Matrix& A_in,
    const Vector& b,
    const Vector& c_in,
    const Vector& d_niz,
    const Vector& d_ver,
    const vector<size_t>& J_baz_input)
{
    ostringstream out;
   // out << fixed << setprecision(6);

    try {
        Matrix A = A_in;
        

        size_t rows = A.size();
        size_t orig_cols = A[0].size();
        Vector c = c_in;
        Vector x(c.size(), 0.0);
        vector<size_t> J_baz = J_baz_input;
        //out << "Phase 1 (nonsence in dual simplex method, only needed to find start basis. Will replace it with manual basis selection l8r)\n";
        //bool criteri = false;

        //Vector omega(rows);
        //for (size_t i = 0; i < rows; ++i) {
        //    double sum = 0.0;
        //    for (size_t j = 0; j < orig_cols; ++j)
        //        sum += A[i][j] * x[j];
        //    omega[i] = b[i] - sum;
        //}

        //size_t art_cols = rows;
        //Vector x_art(art_cols, 0.0);
        //vector<size_t> J_baz;

        //for (size_t i = 0; i < rows; ++i) {
        //    Vector col(rows, 0.0);
        //    if (omega[i] >= 0) {
        //        col[i] = 1.0;
        //        x_art[i] = omega[i];
        //    }
        //    else {
        //        col[i] = -1.0;
        //        x_art[i] = -omega[i];
        //    }
        //    for (size_t r = 0; r < rows; ++r)
        //        A[r].push_back(col[r]);
        //    J_baz.push_back(orig_cols + i);
        //}

        //out << "extended matrix A after adding artificial variables" << endl;
        //for (auto i : A) {
        //    for (auto j : i) {
        //        out << j << " ";

        //    }
        //    out << endl;
        //}
        //out << "c = (0,0,0,0,0,-1,-1,-1)" << endl;
        //Vector d_niz1 = d_niz;
        //Vector d_ver1 = d_ver;
        //for (size_t i = 0; i < art_cols; ++i) {
        //    d_niz1.push_back(0.0);
        //    d_ver1.push_back(numeric_limits<double>::infinity());
        //}

        //x.insert(x.end(), x_art.begin(), x_art.end());
        //for (auto i : x) {
        //    cout << i << " ";
        //}
        //cout<<endl;
        //set<size_t> allinds;
        //for (size_t i = 0; i < orig_cols + art_cols; ++i) allinds.insert(i);

        //Vector c1(orig_cols + art_cols, 0.0);
        //for (size_t i = 0; i < art_cols; ++i) c1[orig_cols + i] = -1.0;
        //int faza1iter = 0;

        //while (!criteri) {
        //    ++faza1iter;
        //    out << "Iteration " << faza1iter << "\n";
        //    Matrix A_baz = selectColumns(A, J_baz);
        //    Vector C_baz = selectElements(c1, J_baz);
        //    Matrix A_baz_inv = inverse(A_baz);
        //    Vector u = multiply(C_baz, A_baz_inv);

        //

        //    vector<size_t> J_nebaz;
        //    set<size_t> J_nebaz_set = allinds;
        //    out << "basis indices (J_b):\n";
        //    for (auto& idx : J_baz) { out << idx << " "; J_nebaz_set.erase(idx); }
        //    out << "\nplan:\n";
        //    for (auto v : x) out << v << " ";
        //    out << "\n";

        //    cout << "basis indices:\n";
        //    for (auto& idx : J_baz) { cout << idx << " "; J_nebaz_set.erase(idx); }
        //    cout << "\nplan:\n";
        //    for (auto v : x) cout << v << " ";
        //    cout << "\n";

        //    out << "u:    ";
        //    for (auto i : u) {
        //        out << i << " ";
        //    }
        //    out << endl;

        //    for (auto& idx : J_nebaz_set) J_nebaz.push_back(idx);
        //    Vector ocenki(orig_cols + art_cols, numeric_limits<double>::infinity());
        //    for (auto& j : J_nebaz) ocenki[j] = c1[j] - dot(getColumn(A, j), u);

        //    int flag = -1;
        //    for (auto j : J_nebaz) {
        //        bool in_bounds = (x[j] >= d_niz1[j] - 1e-12) && (x[j] <= d_ver1[j] + 1e-12);
        //        if ((ocenki[j] <= 0 && x[j] == d_niz1[j]) || (ocenki[j] >= 0 && x[j] == d_ver1[j])) continue;
        //        flag = (int)j; break;
        //    }
        //    if (flag == -1) { criteri = true; break; }

        //   

        //    out << "deltas:  " << endl;
        //    //for (int i = 0; i < ocenki.size(); ++i) {
        //    //    out << "delta_" << i << " = " << ocenki[i] << endl;
        //  //  }
        //    for (auto i : J_nebaz) {
        //        out << "delta_" << i << " = " << ocenki[i] << endl;
        //    }

        //    out << "j_0: " << flag << endl;

        //    Vector l(orig_cols + art_cols);
        //    l[flag] = sgn(ocenki[flag]);

        //    Vector minusAsignOcenka = getColumn(A, flag);
        //    for (auto& i : minusAsignOcenka) {
        //        i *= -sgn(ocenki[flag]);
        //    }
        //    Vector l_baz(J_baz.size());
        //    for (auto& i : J_baz) {
        //        l_baz = multiply(inverse(A_baz), minusAsignOcenka);
        //    }

        //   /* Vector minusAsignOcenka = getColumn(A, flag);
        //    for (auto& v : minusAsignOcenka)
        //        v *= -sgn(ocenki[flag]);

        //    Vector l_baz = multiply(A_baz_inv, minusAsignOcenka);*/

        //    int cnt = 0;
        //    for (auto idx : J_baz)
        //        l[idx] = l_baz[cnt++];


        // //   Vector l(orig_cols + art_cols, 0.0);
        // //   l[flag] = sgn(ocenki[flag]);
        //    for (auto& j : J_nebaz) if (j != flag) l[j] = 0;
        // //   Vector minusAsignOcenka = getColumn(A, flag);
        // //   for (auto& v : minusAsignOcenka) v *= -sgn(ocenki[flag]);
        // //   Vector l_baz = multiply(A_baz_inv, minusAsignOcenka);
        // //   int cnt = 0;
        // //   for (auto idx : J_baz) l[idx] = l_baz[cnt++];

        //    Vector tetas(orig_cols + art_cols, numeric_limits<double>::infinity());
        //    for (int j = 0; j < (int)tetas.size(); ++j) {
        //        if (l[j] > 0) tetas[j] = (d_ver1[j] - x[j]) / l[j];
        //        else if (l[j] < 0) tetas[j] = (d_niz1[j] - x[j]) / l[j];
        //    }
        //    double teta_min = numeric_limits<double>::infinity();

        //    cout << "tetas: " << endl;
        //    for (auto i : tetas) {
        //        cout << i << " ";
        //    }
        //    cout << endl;
        //    cout << "l: " << endl;
        //    for (auto i : l) {
        //        cout << i << " ";
        //    }
        //    cout << endl;

        //    out << "thetas: " << endl;
        //    for (int i = 0; i < tetas.size(); ++i) {
        //        out << "theta_" << i << " = " << tetas[i] << endl;
        //    }
        //    out << endl;
        //    out << "l: ";
        //    for (auto i : l) {
        //        out << i << " ";
        //    }
        //    out << endl;

        //    int j_zvezda = -1;
        //    for (int j = 0; j < (int)tetas.size(); ++j) {
        //        if (tetas[j] < teta_min) { teta_min = tetas[j]; j_zvezda = j; }
        //    }
        //    if (j_zvezda == -1) throw runtime_error("Unbounded in phase 1");
        //    out << "min theta idx: " << j_zvezda << endl;

        //    for (auto& v : l) v *= teta_min;
        //    x = add(x, l);

        //    auto it = find(J_baz.begin(), J_baz.end(), (size_t)j_zvezda);
        //    if (it != J_baz.end()) *it = (size_t)flag;
        //    else J_baz[0] = (size_t)flag;
        //}

        //out << "x after phase1:\n";
        //for (auto v : x) out << v << " ";
        //out << "\n";

      //  x.resize(orig_cols);
       // x.pop_back();
       // x.pop_back();
       // x.pop_back();
        Matrix A_phase2 = A_in;
        c = c_in;
        Vector d_niz2(d_niz.begin(), d_niz.begin() + orig_cols);
        Vector d_ver2(d_ver.begin(), d_ver.begin() + orig_cols);
        out << "\n" << endl;
        out << "Dual simplex method!!!!\n";
        out << "Start basis:\n";
        for (auto i : J_baz) {
            out << i << " ";
        }
        out << endl;
       // vector<size_t> new_basis;
       // for (size_t col = 0; col < orig_cols && new_basis.size() < rows; ++col)
        //    new_basis.push_back(col);
       // if (new_basis.size() == rows) J_baz = new_basis;

        bool criteri2 = false;
        set<size_t> allinds2;
        for (size_t i = 0; i < orig_cols; ++i) allinds2.insert(i);

        int faza2iter = 0;
        while (!criteri2) {
            ++faza2iter;
            out << "Iteration " << faza2iter << "\n";
            cout << "Iteration " << faza2iter << "\n";
            sort(J_baz.begin(), J_baz.end());

            vector<size_t> J_nebaz;


            set<size_t> J_nebaz_set = allinds2;
            out << "basis indices(J_b):\n";
            for (auto& idx : J_baz) { out << idx << " "; J_nebaz_set.erase(idx); }
            out << endl;

            for (auto& idx : J_nebaz_set) 
                J_nebaz.push_back(idx);

            sort(J_nebaz.begin(), J_nebaz.end());

            Matrix A_baz = selectColumns(A_phase2, J_baz);
            Matrix A_nebaz = selectColumns(A_phase2, J_nebaz);
            Vector C_baz = selectElements(c, J_baz);
            Matrix A_baz_inv = inverse(A_baz);
            Vector u = multiply(C_baz, A_baz_inv);

            
            
           // out << "\nplan:\n";
            //for (auto v : x) out << v << " ";
            //out << "\n";

            cout << "basis indices:\n";
            for (auto& idx : J_baz) { cout << idx << " "; J_nebaz_set.erase(idx); }
            cout << "non basis indices:\n";
            for (auto& idx : J_nebaz) { cout << idx << " "; }
            cout << endl;

            cout << "\nplan:\n";
            for (auto v : x) cout << v << " ";
            cout << "\n";
           
            cout << "u:    ";
            for (auto i : u) {
                cout << i << " ";
            }
            cout << endl;

            out << "u(dual plan):"<<endl;
            for (auto i : u) {
                out << i << " ";
            }
            out << endl;

            
            Vector ocenki(orig_cols, numeric_limits<double>::infinity());
            for (auto& j : J_nebaz) ocenki[j] = c[j] - dot(getColumn(A_phase2, j), u);


            cout << "deltas:  " << endl;

            for (auto i : J_nebaz) {
                cout << "delta_" << i << " = " << ocenki[i] << endl;
            }

            out << "deltas (so-called non-basis KOPLAN):  " << endl;

            for (auto i : J_nebaz) {
                out << "delta_" << i << " = " << ocenki[i] << endl;
            }
            Vector kappa(orig_cols);
            for (auto i : J_nebaz) {
                if (ocenki[i] < 0) {
                    kappa[i] = d_niz2[i];
                }
                else if (ocenki[i] >= 0) {
                    kappa[i] = d_ver2[i];
                }
            }
           
            Vector kappa_nebaz(J_nebaz.size());
            int counter = 0;
            for (auto i : J_nebaz) {
                kappa_nebaz[counter++] = kappa[i];
            }
           // Matrix A_nebaz = selectColumns(A_phase2, J_nebaz);
            //Matrix A_baz_inv = inverse(A_baz);
            Vector Aneaun = multiply(A_nebaz, kappa_nebaz);
            Vector kappa_baz(J_baz.size());
            for (auto& i : Aneaun) {
                
                    i*=-1;
                
            }
            Vector b_minus_aneaun = add(b, Aneaun);
            kappa_baz = multiply(A_baz_inv, b_minus_aneaun);
            int cnt = 0;
            for (auto i : J_baz) {
                kappa[i] = kappa_baz[cnt];
                ++cnt;
            }
            bool flag2 = 1;
            for (auto i : J_baz) {
                if (kappa[i] <= d_ver2[i] && kappa[i] >= d_niz2[i]) {
                    continue;
                }
                else {
                    flag2 = 0;
                }
            }
            if (flag2) {
                criteri2 = 1;
                out << "kappa (real plan now):" << endl;
                for (auto i : kappa) {
                    out << i << " ";
                }
                out << endl;
                x = kappa;
                break;
            }
            cout << "kappa" << endl;
            for (auto i : kappa) {
                cout << i << " ";
            }
            cout << endl;

            out << "kappa (pseudo-plan):" << endl;
            for (auto i : kappa) {
                out << i << " ";
            }
            out << endl;

            int j_zvezda = -1;
           // double min_ro = INT_MAX;
            cout << "here1" << endl;
           /* float eps = std::numeric_limits<float>::epsilon();

            Vector ro(J_baz.size());
            for (size_t idx = 0; idx < J_baz.size(); ++idx) {
                size_t i = J_baz[idx];
                if (x[i] < d_niz[i] - eps) {
                    ro[idx] = d_niz[i] - kappa[i];   
                }
                else if (x[i] > d_ver[i] + eps) {
                    ro[idx] = kappa[i] - d_ver[i];  
                }
                else {
                    ro[idx] = 0;
                }
            }

            double max_ro = 0.0;
            int j_zvezda_idx = -1;
            for (size_t idx = 0; idx < ro.size(); ++idx) {
                if (ro[idx] > max_ro) {
                    max_ro = ro[idx];
                    j_zvezda_idx = idx;
                }
            }

            if (j_zvezda_idx == -1) {
                criteri = true;
                break;
            }

            j_zvezda = J_baz[j_zvezda_idx];*/

            //double max_ro = 0;
            //float epsilon = std::numeric_limits<float>::epsilon();
            //for (auto i : J_baz) {
            //   // double ro_jitoe = fabs(kappa[i] - x[i]);
            //    if (fabs(kappa[i] - x[i]) > max_ro && fabs(kappa[i] - x[i]) > epsilon) {
            //        max_ro = fabs(kappa[i] - x[i]);
            //        j_zvezda = i;
            //    }
            //}

            double max_ro = 0.0;
             j_zvezda = -1;
             out << "basis components distances:" << endl;
            for (auto i : J_baz) {
                double ro = distance_to_segment(kappa[i], d_niz[i], d_ver[i]);
                out << "rho_" << i << " = " << ro << endl;
                if (ro > max_ro) {
                    max_ro = ro;
                    j_zvezda = i;
                }
            }

            if (j_zvezda == -1) {
                criteri2 = true;
                break;
            }
            
            cout << "max ro: " << max_ro << endl;
            cout << "j_zvezda: " << j_zvezda << endl;
            out << "max rho (max distance to acceptable values among all kappas): " << max_ro << endl;
            out << "j_*: " << j_zvezda << endl;
            Vector l_u(orig_cols);
            Vector l_u_baz(J_baz.size());
            /*Vector prav_chast(3,0);
            prav_chast[j_zvezda] = -sgn(kappa[j_zvezda] - x[j_zvezda]);
            l_u_baz = multiply(A_baz_inv, prav_chast);*/

            Vector prav_chast(J_baz.size(), 0);
            auto it = find(J_baz.begin(), J_baz.end(), j_zvezda);
            if (it != J_baz.end()) {
                size_t idx_in_basis = distance(J_baz.begin(), it);
                prav_chast[idx_in_basis] = -sgn(kappa[j_zvezda] - x[j_zvezda]);
            }
            else {
                throw runtime_error("j_zvezda not in basis");
            }
            cout << "prav_chast" << endl;
            for (auto i : prav_chast) {
                cout << i << " ";
            }
            cout << endl;
            //l_u_baz = multiply(inverse(transpose(A_baz)), prav_chast);
            l_u_baz = multiply(prav_chast, inverse(A_baz));

            cout << "here2" << endl;
            for (auto i : J_nebaz) {
                l_u[i] = -dot(getColumn(A_phase2, i), l_u_baz);
            }
            cout << "l_u" << endl;
            for (auto i : l_u) {
                cout << i << " ";
            }
            cout << endl;
            cnt = 0;
            for (auto i : J_baz) {
                l_u[i] = l_u_baz[cnt];
                ++cnt;
            }
            cout << "here4" << endl;
            cout << "l_u" << endl;
            for (auto i : l_u) {
                cout << i << " ";
            }
            cout << endl;

            out << "l_u:" << endl;
            for (auto i : l_u) {
                out << i << " ";
            }
            out << endl;

            Vector sigmas(orig_cols, numeric_limits<double>::infinity());
            for (auto i : J_nebaz) {
                if (l_u[i] * ocenki[i] < 0) {
                    sigmas[i] = -ocenki[i] / l_u[i];
                }
            }

            cout << "sigmas" << endl;
            for (auto i : sigmas) {
                cout << i << " ";
            }
            cout << endl;

            out << "sigmas: " << endl;
            for (auto i : sigmas) {
                out << i << " ";
            }
            out << endl;

            bool flag3 = 0;
            int j_1 = -1;
            double sigma_min = numeric_limits<double>::infinity();
            for (auto i : allinds2) {
                if (sigmas[i] < sigma_min) {
                    sigma_min = sigmas[i];
                    flag3 = 1;
                    j_1 = i;
                }
            }
            if (!flag3) {
                out << "no solutions" << endl;
                break;

            }
            cout << "j1: " << j_1 << endl;
            out << "j_1 (min sigma idx): " << j_1 << endl;

            for (auto& idx : J_baz) {
                if (idx == j_zvezda) {
                    idx = j_1;
                    break;
                }
            }

        }

        double ans = dot(c, x);
        out << "\nResult x:\n";
        for (auto v : x) out << v << " ";
        out << "\nObjective value: " << ans << "\n";
    }
    catch (exception& e) {
        out << "Error: " << e.what() << "\n";
    }
    return out.str();
}


// ------------------ C wrapper to be called from JS (ccall) ------------------
extern "C" {
    const char* simplex_run(const char* input_json) {
        try {

            json j = json::parse(input_json);
            Matrix A = j["A"].get<Matrix>();         // expects 3x5
            Vector b = j["b"].get<Vector>();
            Vector c = j["c"].get<Vector>();
            Vector d_niz = j["d_niz"].get<Vector>();
            Vector d_ver = j["d_ver"].get<Vector>();
            //Vector x_start = j["x_start"].get<Vector>(); // expects length 8 (start for extended A)

            //string out = runSimplex(A, b, c, d_niz, d_ver, x_start);

            vector<size_t> J_baz = j["J_baz"].get<vector<size_t>>();

            string out = runSimplex(A, b, c, d_niz, d_ver, J_baz);
            return strdup(out.c_str());
        }
        catch (exception& e) {
            string err = string("JSON/Error: ") + e.what();
            return strdup(err.c_str());
        }
    }
} // extern "C"
