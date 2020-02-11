package question.solution2;

public class S5 {

    public static void main(String[] args) {
        S5 s = new S5();
        TreeNode tree = s.reConstructBinaryTree(
                new int[]{1, 2, 4, 7, 3, 5, 6, 8}, new int[]{4, 7, 2, 1, 5, 3, 8, 6});
    }

    /**
     * 题目描述
     * 输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。
     * 假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
     * 例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
     *
     * @param pre 先序遍历
     * @param in  中序遍历
     * @return 重建的二叉树
     * <p>
     * 1：
     * 4721 | 5386
     * <p>
     * 2：
     * 47 | 2
     * <p>
     * 4：
     * 4|7
     * <p>
     * -》
     * 3：
     * 53|86
     * <p>
     * 3：
     * 5：3
     * <p>
     * 6：
     * 8|6
     */
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        if (pre.length == 0 || pre.length != in.length) {
            return null;
        }

        return reConstructBinaryTree(pre, in, 0, 0, in.length - 1);
    }

    public TreeNode reConstructBinaryTree(int[] pre, int[] in, int startPre, int startIn, int endIn) {
        if (startPre > pre.length - 1 || startIn > endIn) {
            return null;
        }

        int val = pre[startPre];
        TreeNode tree = new TreeNode(val);
        for (int i = startIn; i <= endIn; i++) {
            //在中序遍历中找出当前跟结点
            if (val == in[i]) {
                tree.left = reConstructBinaryTree(pre, in, startPre + 1, startIn, i - 1);
                tree.right = reConstructBinaryTree(pre, in, startPre + i - startIn + 1, i + 1, endIn);
            }
        }

        return tree;
    }

    public static class TreeNode {
        public TreeNode left;
        public TreeNode right;
        int val;

        public TreeNode(int x) {
            val = x;
        }
    }
}