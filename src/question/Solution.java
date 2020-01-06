package question;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Stack;

public class Solution {

    /**
     * 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。
     */
    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();

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

    public void push(int node) {
        stack1.add(node);
    }

    public int pop() {
        if (stack2.isEmpty() && !stack1.isEmpty()) {
            while (!stack1.isEmpty()) {
                stack2.add(stack1.pop());
            }
        }

        if (!stack2.isEmpty()) {
            return stack2.pop();
        }
        return -1;
    }

    /**
     * 题目描述
     * 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
     * 输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
     * 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
     * NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
     *
     * @param array 数组
     * @return 最小值
     */
    public int minNumberInRotateArray(int[] array) {
        int len = array.length;
        if (len == 0) {
            return 0;
        }

        int left = 0;
        int right = len - 1;
        int mid = left;

        while (array[left] >= array[right]) {
            if (right - left == 1) {
                mid = right;
                break;
            }

            mid = left + (right - left) / 2;

            if (array[left] == array[right] && array[left] == array[mid]) {
                //顺序查找,122222->222212或221222,这两种情况无法判断，需要顺序查找
                for (int i = 0; i <= right; i++) {
                    if (array[i] < array[mid]) {
                        mid = i;
                    }
                }
                break;
            }

            if (array[mid] >= array[left]) {
                left = mid;
            } else {
                right = mid;
            }
        }

        return array[mid];
    }

    /**
     * 题目描述
     * 大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。
     * n<=39
     *
     * @param n
     * @return
     */
    public long Fibonacci(int n) {
        if (n < 2) {
            return n;
        }
        long a = 0;
        long b = 1;
        long m = b;
        for (int i = 2; i <= n; i++) {
            m = a + b;
            a = b;
            b = m;
        }
        return m;
    }

    /**
     * 题目描述
     * 一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
     *
     * @param n
     * @return
     */
    public int JumpFloor(int n) {
        if (n <= 2) {
            return n;
        }
        int a = 1;
        int b = 2;
        int m = b;
        for (int i = 3; i <= n; i++) {
            m = a + b;
            a = b;
            b = m;
        }
        return m;
    }

    /**
     * 题目描述
     * 一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
     * 链接：https://www.nowcoder.com/questionTerminal/22243d016f6b47f2a6928b4313c85387?f=discussion
     * 来源：牛客网
     * <p>
     * f(n-1) = f(0) + f(1)+f(2)+f(3) + ... + f((n-1)-1) = f(0) + f(1) + f(2) + f(3) + ... + f(n-2)
     * <p>
     * f(n) = f(0) + f(1) + f(2) + f(3) + ... + f(n-2) + f(n-1) = f(n-1) + f(n-1)
     * <p>
     * 可以得出：
     * <p>
     * f(n) = 2*f(n-1)
     *
     * @param number
     * @return
     */
    int jumpFloorII(int number) {
        if (number <= 2) {
            return number;
        }

        int sum = 0;
        int a = 1;
        int b = 2;
        for (int i = 3; i <= number; i++) {
            sum = 2 * b;
            b = sum;
            a = b;
        }
        return sum;
    }

    /**
     * 题目描述
     * 我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
     *
     * @param target
     * @return
     */
    public int RectCover(int target) {
        if (target <= 2) {
            return target;
        }

        int sum = 0;
        int a = 1;
        int b = 2;
        for (int i = 3; i <= target; i++) {
            sum = b + a;
            a = b;
            b = sum;
        }
        return sum;
    }

    /**
     * 题目描述
     * 输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
     *
     * @param n
     * @return
     */
    public int NumberOf1(int n) {
        int count = 0;
        while (n != 0) {
            if ((n & 1) == 1) {
                count++;
            }
            n = n >>> 1; //>>>无视符号位右移，左边补0。>>右移，符号位负数补1，整数补0，会有问题
        }
        return count;
    }

    /**
     * 题目描述
     * 给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
     * <p>
     * 保证base和exponent不同时为0
     *
     * @param base
     * @param exponent
     * @return
     */
    public double Power(double base, int exponent) {
        if (exponent == 0) {
            return 1;
        }

        if (base == 0.0 && exponent < 0) {
            return 0; // 对0求倒数没有意义
        }

        if (exponent < 0) {
            return 1 / getPower(base, -exponent);
        } else {
            return getPower(base, exponent);
        }

    }

    public double getPower(double base, int exponent) {
        if (exponent == 0) {
            return 1;
        }
        if (exponent == 1) {
            return base;
        }

        double ret = getPower(base, exponent >> 1);
        ret *= ret;
        if ((exponent & 0x1) == 1) {
            ret *= base;
        }
        return ret;
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
