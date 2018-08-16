#include <string>
#include <unordered_set>
#include <iostream>
#include <functional>

using namespace std;

struct Base { 
    public:
    virtual int get_m1() const = 0;
    virtual int get_m2() const = 0;
    virtual string toString() const = 0; 
};

struct TwoInt : public Base {
    public:
    TwoInt(int d1, int d2): m1(d1), m2(d2) {}
    string toString() const override;
    int get_m1() const override { return m1; }
    int get_m2() const override { return m2; }

    private:
    int m1;
    int m2;
};

string TwoInt::toString() const {
    return std::to_string(m1) + "/" + std::to_string(m2);
}

struct Hasher {
    size_t operator()(Base* const pB) const {
        auto seed = pB->get_m2();
        return hash<int>()(pB->get_m1()) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    }
};

struct BasePtrComp {
    bool operator()(Base* const p1, Base* const p2) const {
        return p1->get_m1() == p2->get_m1() && p1->get_m2() == p2->get_m2();
    }
};

int main() {
    using PairSet = unordered_set<Base*, Hasher, BasePtrComp>;
    Base* p1 = new TwoInt(3, 5); cout << "p1 @ " << p1 << endl;
    Base* p2 = new TwoInt(4, 5); cout << "p2 @ " << p2 << endl;
    Base* p3 = new TwoInt(3, 5); cout << "p3 @ " << p3 << endl;

    PairSet Cont(16, Hasher(), BasePtrComp());
    auto resl1 =  Cont.insert(p1);
    auto resl2 =  Cont.insert(p2);
    auto resl3 =  Cont.insert(p3);
    cout << *(resl3.first) << "," << resl3.second << endl;
    cout << Cont.size() << endl;
    for (auto el: Cont) {
        cout << el << endl;
    }
}
