class Solution {
public:
    /*
     * @param root: the root of binary tree
     * @param target: An integer
     * @return: all valid paths
     */
    vector<vector<int>> binaryTreePathSum(TreeNode * root, int target) {
        // write your code here
        vector<vector<int>> result;
        vector<int> path;
        if (root == NULL) {
            return result;
        } 
        dfs(root, result, path, target);
        return result;
    }
    
private:
    class DFSGuard {
    public:
        DFSGuard(vector<int>& apath, int addition): path(apath) {
            path.push_back(addition);
        }
        ~DFSGuard() {
            path.pop_back();
        } 
    private:
        vector<int>& path;        
    };
    
    void dfs(TreeNode* root, vector<vector<int>>& result, vector<int>& path, int target) {  
        if (root->left == nullptr && root->right == nullptr) {
            // leaf 
            if (target == root->val) {
                DFSGuard guard(path, root->val); 
                result.push_back(path);  
            }
            return;
        }
        if (root->left) {
            DFSGuard guard(path, root->val); 
            dfs(root->left, result, path, target - root->val); 
        }
        if (root->right) {
            DFSGuard guard(path, root->val); 
            dfs(root->right, result, path, target - root->val); 
        } 
    }
};
