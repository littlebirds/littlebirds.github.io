#include "trie.hpp"
#include <vector>
#include <iostream>

void Trie::insert(const std::string& word) {
    Node* pNode = &root;
    for (auto c : word) { 
        pNode = &(pNode->children[c]);
    }
    pNode->isEndOfWord = true;
}

bool Trie::contains(const std::string& word) const {
    const Node* p = &root;
    for (auto c: word) {
        auto it = p->children.find(c);
        if (it == p->children.end()) {
            return false;
        }
        p = &(it->second);
    }
    return p->isEndOfWord;
} 

void Trie::print(const Trie::Node& root, std::string& path) const {
    if (root.isEndOfWord) {
        std::cout << path << std::endl; 
    }
    for (auto& kv: root.children) { 
        char prefix = kv.first;
        path.push_back(prefix);
        const Node& pNode = (kv.second);
        print(pNode, path);
        path.pop_back();
    }
}

void Trie::display() const {
    std::string word("");
    this->print(root, word);
}

int main() {
    std::vector<std::string> words {"foo", "bar", "baz", "barz", "sudo"};
    Trie trie;
    for (auto& word: words) {
        trie.insert(word);
    }
    trie.display();
}
