package question.solution2;

public class S6 {
    /**
     * 二叉树的下一个结点
     * 题目描述
     * 给定一个二叉树和其中的一个结点，请找出"中序遍历"顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。
     *
     * @param pNode 二叉树结点
     * @return 二叉树的下一个结点
     */
    public TreeLinkNode GetNext(TreeLinkNode pNode) {
        if (pNode == null) {
            return null;
        }

        if (pNode.right != null) {
            pNode = pNode.right;
            while (pNode.left != null) {
                pNode = pNode.left;
            }
            return pNode;
        }

        //注意细节
        while (pNode.next != null) {
            if (pNode.next.left == pNode) {
                return pNode.next;
            } else {
                pNode = pNode.next;
            }
        }
        return null;
    }

    public class TreeLinkNode {
        int val;
        TreeLinkNode left = null;
        TreeLinkNode right = null;
        TreeLinkNode next = null;

        TreeLinkNode(int val) {
            this.val = val;
        }
    }
}
