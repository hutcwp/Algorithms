package question;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Stack;

public class Solution {

    /**
     * 题目描述
     * 输入一个链表，按链表从尾到头的顺序返回一个ArrayList。
     * 如果只是打印出值的话，则只需要用递归栈即可
     *
     * @param listNode 链表
     * @return 集合
     */
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        ArrayList<Integer> list = new ArrayList<>();
        while (listNode != null) {
            list.add(listNode.val);
            listNode = listNode.next;
        }
        int pre = 0;
        int end = list.size() - 1;
        while (pre < end) {
            int t = list.get(pre);
            list.set(pre, list.get(end));
            list.set(end, t);
            pre++;
            end--;
        }
        return list;
    }

    public void printTree(TreeNode tree) {
        if (tree == null) {
            return;
        }

        System.out.println(tree.val);
        printTree(tree.left);
        printTree(tree.right);

    }

    /**
     * 题目描述
     * 在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，
     * 每一列都按照从上到下递增的顺序排序。请完成一个函数，
     * 输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
     *
     * @param target 目标值
     * @param array  数组
     * @return 是否存在
     */
    public boolean Find(int target, int[][] array) {
        boolean result = false;
        // 注意这里的行列取值，要对应上数组定义
        int colSize = array[0].length; //行
        int rowSize = array.length; //列
        int col = 0;
        int row = rowSize - 1;

        while (col < colSize && row >= 0) {
            int cur = array[row][col];
            if (cur == target) {
                result = true;
                break;
            } else if (target > cur) {
                col++;
            } else {
                row--;
            }
        }

        return result;
    }

    /**
     * 题目描述
     * 请实现一个函数，将一个字符串中的每个空格替换成“%20”。
     * 例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
     *
     * @param str 传入的字符串
     * @return 处理后的字符串
     */
    public String replaceSpace(StringBuffer str) {
        int spaceNum = 0;
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == ' ') {
                spaceNum++;
            }
        }

        int oldLen = str.length(); //需要先记录下，后面重新设置后会变化
        int newLen = str.length() + spaceNum * 2;
        str.setLength(newLen);
        newLen--;
        for (int i = oldLen - 1; i >= 0; i--) {
            char c = str.charAt(i);
            if (c == ' ') {
                str.setCharAt(newLen--, '0');
                str.setCharAt(newLen--, '2');
                str.setCharAt(newLen--, '%');
            } else {
                str.setCharAt(newLen--, c);
            }
        }
        return str.toString();
    }

    /**
     * 二叉树前序遍历
     *
     * @param root 根节点
     * @return 先序遍历的集合
     */
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }

        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.pollFirst();

            res.add(node.val);
            if (node.right != null) {
                queue.addFirst(node.right);
            }
            if (node.left != null) {
                queue.addFirst(node.left);
            }
        }
        return res;
    }

    /**
     * 二叉树中序遍历
     *
     * @param root 根结点
     * @return 中序遍历的集合
     */
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }

        Stack<TreeNode> stack = new Stack<>();
        while (root != null || !stack.isEmpty()) {
            if (root != null) {
                stack.add(root);
                root = root.left;
            } else {
                root = stack.pop();
                res.add(root.val);
                root = root.right;
            }
        }
        return res;
    }

    /**
     * 二叉树后序遍历
     *
     * @param root 根结点
     * @return 后序遍历集合
     */
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }

        Stack<TreeNode> s1 = new Stack<>();
        Stack<TreeNode> s2 = new Stack<>();
        s1.add(root);
        while (!s1.empty()) {
            TreeNode node = s1.pop();
            s2.add(node);
            if (node.left != null) {
                s1.add(node.left);
            }
            if (node.right != null) {
                s1.add(node.right);
            }
        }

        while (!s2.empty()) {
            res.add(s2.pop().val);
        }

        return res;
    }

    /**
     * 给出前序遍历和中序遍历，重键二叉树
     *
     * @param pre 先序遍历的顺序
     * @param in  中序遍历的顺序
     * @return 重构后的二叉树
     */
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        if (pre.length == 0 || in.length == 0 || pre.length != in.length) {
            return null;
        }

        return createTrees(pre, in, 0, 0, in.length - 1);
    }

    private TreeNode createTrees(int[] pre, int[] in, int pre_start, int in_start, int in_end) {
        if (pre_start < 0 || pre_start >= pre.length) {
            return null;
        }

        if (in_start < 0 || in_end >= in.length || in_start > in_end) {
            return null;
        }

        int rootVal = pre[pre_start];
        TreeNode root = new TreeNode(rootVal);
        int mid = in_start;
        for (int i = in_start; i <= in_end; i++) {
            if (in[i] == rootVal) {
                mid = i;
                break;
            }
        }

        root.left = createTrees(pre, in, pre_start + 1, in_start, mid - 1);
        root.right = createTrees(pre, in, mid - in_start + pre_start + 1, mid + 1, in_end);
        return root;
    }

    // Definition for binary tree
    public static class TreeNode {
        public TreeNode left;
        public TreeNode right;
        int val;

        public TreeNode(int x) {
            val = x;
        }
    }

    /**
     * 链表结构
     */
    public class ListNode {
        public int val;
        public ListNode next = null;

        public ListNode(int val) {
            this.val = val;
        }
    }

}
