#include <memory>
#include <string>
#include <map>
#include <vector>

class Trie {
private:
    struct Node {
        bool isEndOfWord;
        std::map<char, Node> children;  
        // ctor
        Node(): isEndOfWord(false) {}      
    };  
    Node root;  
private:
    void print(const Node& root, std::string& path) const; 
public:
    void insert(const std::string& word);
    bool contains(const std::string& word) const;  
    void display( ) const;
}; 
