package question.solution1;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
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

    /**
     * 大数相加，num1和num2都大于等于0，如果一方小于0就变成相减了。
     *
     * @param num1
     * @param num2
     * @return
     */
    public String addStrings(String num1, String num2) {
        Stack<Integer> res = new Stack<>();
        int len1 = num1.length() - 1;
        int len2 = num2.length() - 1;
        int flag = 0;
        while (len1 >= 0 || len2 >= 0 || flag > 0) {
            int sum = 0;
            if (len1 >= 0) {
                sum += num1.charAt(len1--) - '0';
            }
            if (len2 >= 0) {
                sum += num2.charAt(len2--) - '0';
            }
            sum += flag;
            flag = sum / 10;
            sum = sum % 10;
            res.add(sum);
        }

        StringBuilder sb = new StringBuilder();
        while (!res.isEmpty()) {
            sb.append(res.pop());
        }
        return sb.toString();
    }

    /**
     * 大数乘法
     *
     * @param num1
     * @param num2
     * @return
     */
    public String multiply(String num1, String num2) {
        if (num1.equals("0") || num2.equals("0")) {
            return "0";
        }
        // 符号
        boolean op = false;
        if (num1.charAt(0) == '-') {
            op = true;
            num1 = num1.replace("-", "");
        }
        if (num2.charAt(0) == '-') {
            op = !op;
            num2 = num2.replace("-", "");
        }

        char[] res = new char[num1.length() + num2.length()];
        for (int i = 0; i < res.length; i++) {
            res[i] = 'n';
        }

        char[] n1 = num1.toCharArray();
        char[] n2 = num2.toCharArray();

        int flag = 0; //进位
        int cur = 0;//当前位数
        int curres = 0; //当前结果

        for (int i = n2.length - 1; i >= 0; i--) {
            cur = n2.length - i - 1;
            flag = 0;
            for (int j = n1.length - 1; j >= 0; j--) {
                int a = n2[i] - 48;
                int b = n1[j] - 48;
                int sum = a * b + flag; //乘积加上进位
                if (res[cur] != 'n') {
                    sum += (res[cur] - 48);
                }
                curres = sum % 10;
                flag = sum / 10;
                res[cur] = (char) (curres + 48);
                cur++;
            }

            // 处理最后一位进位
            if (flag != 0) {
                res[cur] = (char) (flag + 48);
            }

        }

        // 结果倒置
        StringBuilder sb = new StringBuilder();
        for (int i = res.length - 1; i >= 0; i--) {
            if (res[i] != 'n') {
                sb.append(res[i]);
            }
        }

        //处理符号
        if (op) {
            return "-" + sb.toString();
        }
        return sb.toString();
    }

    /**
     * 题目描述
     * 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，
     * 所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
     * 1 2 4 3 5 6 8
     *
     * @param array
     */
    public void reOrderArray(int[] array) {
        int len = array.length;
        if (len <= 1) {
            return;
        }
        int pre = 0;

        for (int i = 0; i < len; i++) {
            int tmp = array[i];
            if (tmp % 2 == 1) {
                for (int j = 0; j < i; j++) {
                    if (array[j] % 2 == 0) {

                    }
                }
            }
        }
    }

    /**
     * 倒数第k个链表
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode FindKthToTail(ListNode head, int k) {
        if (head == null || k < 0) {
            return null;
        }

        ListNode cur = head;
        int num = k;
        while (num > 0 && head != null) {
            head = head.next;
            num--;
        }

        while (head != null) {
            head = head.next;
            cur = cur.next;
        }

        if (num > 0) {
            //k>n
            return null;
        }
        return cur;

    }

    /**
     * 题目描述
     * 输入一个链表，反转链表后，输出新链表的表头。
     * 1   | 2   | 3
     * p1  | p2  | head
     *
     * @param head
     * @return
     */
    public ListNode ReverseList(ListNode head) {
        if (head == null) {
            return null;
        }

        ListNode p1;
        ListNode p2 = null;
        while (head != null) {
            p1 = p2;
            p2 = head;
            head = head.next;
            p2.next = p1;
        }

        return p2;
    }

    /**
     * 题目描述
     * 输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
     *
     * @param list1
     * @param list2
     * @return
     */
    public ListNode Merge(ListNode list1, ListNode list2) {
        if (list1 == null) {
            return list2;
        }
        if (list2 == null) {
            return list1;
        }

        ListNode head = null;
        if (list1.val <= list2.val) {
            head = list1;
            head.next = Merge(list1.next, list2);
        } else {
            head = list2;
            head.next = Merge(list1, list2.next);
        }

        return head;
    }

    /**
     * 题目描述
     * 输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
     *
     * @param root1
     * @param root2
     * @return
     */
    public boolean HasSubtree(TreeNode root1, TreeNode root2) {
        //注意细节root2 == null，不判断后面会报空指针
        if (root2 == null || root1 == null) {
            return false;
        }


        return isSubtree(root1, root2) || HasSubtree(root1.left, root2) || HasSubtree(root1.right, root2);
    }

    public boolean isSubtree(TreeNode root1, TreeNode root2) {
        //注意，因为是子结构，当root2==null时，则判断结束
        if (root2 == null) {
            return true;
        }

        if (root1 != null) {
            if (root1.val != root2.val) {
                return false;
            } else {
                return isSubtree(root1.left, root2.left) && isSubtree(root1.right, root2.right);
            }

        }
        return false;
    }

    /**
     * 题目描述
     * 操作给定的二叉树，将其变换为源二叉树的镜像。
     *
     * @param root
     */
    public void Mirror(TreeNode root) {
        if (root == null) {
            return;
        }

        TreeNode tmp = root.left;
        root.left = root.right;
        root.right = tmp;
        Mirror(root.left);
        Mirror(root.right);
    }

    /**
     * 题目描述
     * 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，
     * 例如，如果输入如下4 X 4矩阵：
     * 1  2  3  4
     * 5  6  7  8
     * 9  10 11 12
     * 13 14 15 16
     * 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
     *
     * @param matrix
     * @return
     */
    public ArrayList<Integer> printMatrix(int[][] matrix) {
        ArrayList<Integer> res = new ArrayList<>();
        if (matrix.length == 0) {
            return res;
        }

        int left = 0;
        int top = 0;
        int right = matrix[0].length - 1;
        int bottom = matrix.length - 1;

        int col = right;
        int row = bottom;
        // 细节是魔鬼
        while (left * 2 < col + 1 && top * 2 < row + 1) {
            for (int i = left; i <= right; i++) {
                res.add(matrix[top][i]);
                System.out.println(matrix[top][i]);
            }

            if (top < bottom) {
                for (int i = top + 1; i <= bottom; i++) {
                    res.add(matrix[i][right]);
                    System.out.println(matrix[i][right]);
                }
            }


            if (top < bottom && left < right) {
                for (int i = right - 1; i >= left; i--) {
                    res.add(matrix[bottom][i]);
                    System.out.println(matrix[bottom][i]);
                }
            }


            if (top < bottom - 1 && left < right) {
                for (int i = bottom - 1; i > top; i--) {
                    res.add(matrix[i][left]);
                    System.out.println(matrix[i][left]);
                }
            }

            left++;
            right--;
            top++;
            bottom--;
        }

        return res;
    }

    /**
     * 题目描述
     * 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。
     * 假设压入栈的所有数字均不相等。
     * 例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，
     * 但4,3,5,1,2就不可能是该压栈序列的弹出序列。
     * （注意：这两个序列的长度是相等的）
     *
     * @param pushA
     * @param popA
     * @return
     */
    public boolean IsPopOrder(int[] pushA, int[] popA) {
        int lenA = pushA.length;
        int lenB = popA.length;

        if (lenA != lenB) {
            return false;
        }
        if (lenA == 0) {
            return true;
        }

        Stack<Integer> s = new Stack<>();

        int j = 0;
        for (int i = 0; i < lenA; i++) {
            //如果入栈元素等于出栈元素，就忽略，否则入栈s
            if (pushA[i] == popA[j]) {
                j++;
            } else {
                s.push(pushA[i]);
            }
        }

        // s非空，一个一个抛出来比对
        while (!s.isEmpty() && j < lenB) {
            if (s.peek() == popA[j]) {
                s.pop();
                j++;
            } else {
                //如果栈顶元素不等于出栈序列当前元素即结束
                break;
            }
        }

        return s.isEmpty();
    }

    /**
     * 题目描述
     * 从上往下打印出二叉树的每个节点，同层节点从左至右打印。
     * 层次遍历
     *
     * @param root
     * @return
     */
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }

        Queue<TreeNode> queue = new LinkedList<>();

        queue.add(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.peek();
            queue.remove();
            res.add(node.val);
            if (node.left != null) {
                queue.add(node.left);
            }
            if (node.right != null) {
                queue.add(node.right);
            }
        }
        return res;
    }

    /**
     * 题目描述
     * 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。
     * 假设输入的数组的任意两个数字都互不相同。
     *
     * @param sequence
     * @return
     */
    public boolean VerifySquenceOfBST(int[] sequence) {
        if (sequence.length == 0) {
            return false;
        }

        return isSquenceOfBST(sequence, 0, sequence.length - 1);
    }

    private boolean isSquenceOfBST(int[] sequence, int start, int end) {
        if (start >= end) {
            return true;
        }

        int root = sequence[end];
        int i = start;
        int j = start;
        for (i = start; i < end; i++) {
            if (sequence[i] > root) {
                break;
            }
        }

        j = i; //注意这里赋值，不能放到上面的循环退出条件中。存在右子树为空的情况
        for (; j < end; j++) {
            if (sequence[j] < root) {
                return false;
            }
        }

        boolean left = isSquenceOfBST(sequence, start, i - 1);
        boolean right = isSquenceOfBST(sequence, i, end - 1);
        return left && right;
    }

    /**
     * 题目描述
     * 输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。
     * 路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。
     * (注意: 在返回值的list中，数组长度大的数组靠前)
     *
     * @param root
     * @param target
     * @return
     */
    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int target) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        ArrayList<Integer> path = new ArrayList<>();
        if (root != null) {
            dfsPath(root, target, res, path);
        }

        return res;
    }

    private void dfsPath(TreeNode node, int target, ArrayList<ArrayList<Integer>> list,
                         ArrayList<Integer> path) {
        if (node == null) {
            return;
        }

        path.add(node.val);

        if (node.left == null && node.right == null) {
            if (target - node.val == 0) {
                //细节1
                ArrayList<Integer> r = new ArrayList<>(path);
                list.add(r);
            }
        }

        dfsPath(node.left, target - node.val, list, path);
        dfsPath(node.right, target - node.val, list, path);
        //细节2，Java的list的remove需要使用Integer.valueOf(node.val)
        path.remove(Integer.valueOf(node.val));
    }

    /**
     * 题目描述
     * 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。
     *
     * @param pRootOfTree
     * @return
     */
    TreeNode lastRoot = null;

    public RandomListNode Clone(RandomListNode pHead) {
        if (pHead == null) {
            return null;
        }

        RandomListNode cloneNode;
        RandomListNode cloneHead;
        RandomListNode curNode = pHead;
        //1.复制结点
        while (curNode != null) {
            cloneNode = new RandomListNode(curNode.label);
            cloneNode.next = curNode.next;
            curNode.next = cloneNode;
            curNode = cloneNode.next;
        }

        //调整随机结点
        curNode = pHead;
        cloneHead = pHead.next;
        while (curNode != null) {
            if (curNode.random != null) {
                curNode.next.random = curNode.random.next;
            }
            curNode = curNode.next.next;
        }

        //分离原结点和新结点
        curNode = pHead;
        cloneNode = cloneHead;
        while (curNode != null) {
            curNode.next = cloneNode.next;
            cloneNode.next = cloneNode.next == null ? null : cloneNode.next.next;
            curNode = curNode.next;
            cloneNode = cloneNode.next;
        }

        return cloneHead;
    }

    public TreeNode Convert(TreeNode pRootOfTree) {

        if (pRootOfTree == null) {
            return null;
        }

        if (pRootOfTree.left == null && pRootOfTree.right == null) {
            return pRootOfTree;
        }

        TreeNode left = Convert(pRootOfTree.left);
        left.right = lastRoot;


        TreeNode right = Convert(pRootOfTree.right);
        lastRoot.right = right;


        return left == null ? right : left;

    }

    /**
     * 题目描述
     * 输入一个字符串,按字典序打印出该字符串中字符的所有排列。
     * 例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
     * 输入描述:
     * 输入一个字符串,长度不超过9(可能有字符重复),字符只包括大小写字母。
     *
     * @param str
     * @return
     */
    public ArrayList<String> Permutation(String str) {
        ArrayList<String> res = new ArrayList<>();
        if (str == null || str.length() == 0) {
            return res;
        }

        nextStr(res, 0, str.toCharArray());
        Collections.sort(res);
        return res;
    }

    public void nextStr(ArrayList<String> res, int cur, char c[]) {
        if (cur == c.length - 1) {
            res.add(String.valueOf(c));
        }

        Set<Character> charSet = new HashSet<>();
        for (int i = cur; i < c.length; i++) {
            if (cur == i || !charSet.contains(c[i])) {
                charSet.add(c[i]);
                swap(cur, i, c);
                nextStr(res, cur + 1, c);
                swap(cur, i, c);
            }
        }
    }

    private void swap(int start, int end, char c[]) {
        char tmp = c[start];
        c[start] = c[end];
        c[end] = tmp;
    }

    /**
     * 题目描述
     * 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。
     * 由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
     *
     * @param array
     * @return
     */
    public int MoreThanHalfNum_Solution(int[] array) {
        if (array.length == 0) {
            return 0;
        }

        int num = array[0];
        int count = 1;
        for (int i = 1; i < array.length; i++) {
            if (num != array[i]) {
                count--;
            } else {
                count++;
            }

            if (count == 0) {
                num = array[i];
                count = 1;
            }
        }

        count = 0;
        for (int anArray : array) {
            if (anArray == num) {
                count++;
            }
        }
        return count > (array.length / 2) ? num : 0;
    }

    /**
     * 题目描述
     * 输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
     *
     * @param input
     * @param k
     * @return
     */
    public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
        ArrayList<Integer> res = new ArrayList<>();
        if (input.length < k || k <= 0) {
            return res;
        }

        PriorityQueue<Integer> queue = new PriorityQueue<>(Comparator.reverseOrder());
        for (int item : input) {
            if (queue.size() < k) {
                queue.add(item);
            } else {
                // max heap
                queue.size();
                int max = queue.peek();
                if (item < max) {
                    queue.poll();
                    queue.add(item);
                }
            }
        }

        res.addAll(queue);
        return res;
    }

    /**
     * 最大连续子序列的和
     */
    public int FindGreatestSumOfSubArray(int[] array) {
        if (array.length == 1) {
            return array[0];
        }

        int sum = array[0];
        int cur = array[0];
        for (int i = 1; i < array.length; i++) {
            cur += array[i];
            sum = Math.max(sum, cur);
            if (cur < 0) {
                cur = 0;
            }
        }
        return sum;
    }

    /**
     * 题目描述
     * 求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？
     * 为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。
     * ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。
     *
     * @param n
     * @return
     */
    public int NumberOf1Between1AndN_Solution(int n) {
        int count = 0;//1的个数
        int i = 1;//当前位
        int current = 0, after = 0, before = 0;
        while ((n / i) != 0) {
            current = (n / i) % 10; //高位数字
            before = n / (i * 10); //当前位数字
            after = n - (n / i) * i; //低位数字
            //如果为0,出现1的次数由高位决定,等于高位数字 * 当前位数
            if (current == 0) {
                count += before * i;
            }
            //如果为1,出现1的次数由高位和低位决定,高位*当前位+低位+1
            else if (current == 1) {
                count += before * i + after + 1;
            }
            //如果大于1,出现1的次数由高位决定,//（高位数字+1）* 当前位数
            else {
                count += (before + 1) * i;
            }
            //前移一位
            i = i * 10;
        }
        return count;
    }

    /**
     * 题目描述
     * 输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
     * 例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。
     *
     * @param numbers
     * @return
     */
    public String PrintMinNumber(int[] numbers) {
        int n;
        StringBuilder result = new StringBuilder();
        ArrayList<Integer> list = new ArrayList<>();
        n = numbers.length;
        for (int i = 0; i < n; i++) {
            list.add(numbers[i]);

        }
        list.sort((str1, str2) -> {
            String s1 = str1 + "" + str2;
            String s2 = str2 + "" + str1;
            return s1.compareTo(s2);
        });

        for (int s : list) {
            result.append(s);
        }
        return result.toString();

    }

    /**
     * 题目描述
     * 把只包含质因子2、3和5的数称作丑数（Ugly Number）。
     * 例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。
     *
     * @param index
     * @return
     */
    public int GetUglyNumber_Solution(int index) {
        if (index <= 6) {
            return index;
        }

        int res[] = new int[100000];
        res[0] = 1;
        int cur = 0;
        int num2_index = 0;
        int num3_index = 0;
        int num5_index = 0;
        for (int i = 0; i < index; i++) {
            int tmp_2 = 2 * res[num2_index];
            int tmp_3 = 3 * res[num3_index];
            int tmp_5 = 5 * res[num5_index];
            int num = Math.min(tmp_2, Math.min(tmp_3, tmp_5));
            res[++cur] = num;
            if (tmp_2 == num) {
                num2_index++;
            }
            if (tmp_3 == num) {
                num3_index++;
            }
            if (tmp_5 == num) {
                num5_index++;
            }
        }
        return res[index - 1];
    }

    /**
     * 题目描述
     * 在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）.
     *
     * @param str
     * @return
     */
    public int FirstNotRepeatingChar(String str) {
        char cs[] = new char[256];
        for (int i = 0; i < str.length(); i++) {
            cs[str.charAt(i)]++;
        }

        for (int i = 0; i < str.length(); i++) {
            if (cs[str.charAt(i)] == 1) {
                return i;
            }
        }
        return -1;
    }

    /**
     * 题目描述
     * 输入两个链表，找出它们的第一个公共结点。
     *
     * @param pHead1
     * @param pHead2
     * @return
     */
    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        int lena = 0;
        int lenb = 0;
        ListNode head = pHead1;
        while (head != null) {
            lena++;
            head = head.next;
        }

        head = pHead2;
        while (head != null) {
            lenb++;
            head = head.next;
        }

        int diff = 0;
        if (lena > lenb) {
            diff = lena - lenb;
            while (diff > 0) {
                diff--;
                pHead1 = pHead1.next;
            }

        } else {
            diff = lenb - lena;
            while (diff > 0) {
                diff--;
                pHead2 = pHead2.next;
            }
        }

        while (pHead1 != null && pHead2 != null) {
            if (pHead1 == pHead2) {
                return pHead1;
            }

            pHead1 = pHead1.next;
            pHead2 = pHead2.next;
        }
        return null;
    }

    /**
     * 题目描述
     * 统计一个数字在排序数组中出现的次数。
     * 1 3 4 4 5 6
     *
     * @param array
     * @param k
     * @return
     */
    public int GetNumberOfK(int[] array, int k) {
        if (array.length == 0) {
            return 0;
        }

        int left = getLeftK(array, k);
        int last = getLastK(array, k);
        if (left == -1 || last == -1) {
            return 0;
        }
        return last - left + 1;
    }

    private int getLeftK(int[] array, int k) {
        int left = 0;
        int right = array.length - 1;
        int mid;
        while (left < right) {
            mid = (left + right) / 2;
            int midValue = array[mid];
            if (midValue < k) {
                left = mid + 1;
            } else if (midValue >= k) {
                right = mid;
            }
            if (right - left == 1) {
                break;
            }
        }

        return array[left] == k ? left : array[right] == k ? right : -1;
    }

    private int getLastK(int[] array, int k) {
        int left = 0;
        int right = array.length - 1;
        int mid;
        while (left < right) {
            mid = (left + right) / 2;
            int midValue = array[mid];
            if (midValue <= k) {
                left = mid;
            } else if (midValue > k) {
                right = mid - 1;
            }
            if (right - left == 1) {
                break;
            }
        }

        return array[right] == k ? right : array[left] == k ? left : -1;
    }

    /**
     * 题目描述
     * 输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。
     *
     * @param root
     * @return
     */
    public int TreeDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }

        return Math.max(TreeDepth(root.left), TreeDepth(root.right)) + 1;
    }

    /**
     * 题目描述
     * 输入一棵二叉树，判断该二叉树是否是平衡二叉树。
     *
     * @param root
     * @return
     */
    public boolean IsBalanced_Solution(TreeNode root) {
        if (root == null) {
            return true;
        }

        return getHeight(root) != -1;
    }

    private int getHeight(TreeNode root) {
        if (root == null) {
            return 0;
        }

        int leftHeight = getHeight(root.left);
        int rightHeight = getHeight(root.right);
        //补丁
        if (leftHeight == -1 || rightHeight == -1) {
            return -1;
        }

        if (Math.abs(leftHeight - rightHeight) > 1) {
            return -1;
        } else {
            return Math.max(leftHeight, rightHeight) + 1;
        }
    }

    /**
     * 题目描述
     * 如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，
     * 那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，
     * 那么中位数就是所有数值排序之后中间两个数的平均值。
     * 我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。
     */
    public ArrayList<Integer> list = new ArrayList<>();
    /**
     * 题目描述
     * 请实现一个函数用来找出字符流中第一个只出现一次的字符。
     * 例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。
     * 当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。
     * 输出描述:
     * 如果当前字符流没有存在出现一次的字符，返回#字符。
     *
     * @param ch
     */
    int stream[] = new int[256];
    int curIndex = 1;
    /**
     * 题目描述
     * 给定一棵二叉搜索树，请找出其中的第k小的结点。例如，（5，3，7，2，4，6，8）中，按结点数值大小顺序第三小结点的值为4。
     *
     * @param pRoot
     * @param k
     * @return
     */
    int cur = 0;
    private int index = -1; //细节，注意这里为-1。

    /**
     * 题目描述
     * 一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。
     *
     * @param array
     * @param num1
     * @param num2
     */
    public void FindNumsAppearOnce(int[] array, int num1[], int num2[]) {
        if (array.length < 2) {
            return;
        }

        int temp = 0;
        for (int anArray : array) {
            temp = temp ^ anArray;
        }

        int index = 0;
        while ((temp & 1) == 0) {
            temp = temp >> 1;
            ++index;
        }

        int num_1 = 0;
        int num_2 = 0;
        for (int anArray : array) {
            if (isBit0(anArray, index)) {
                num_1 ^= anArray;
            } else {
                num_2 ^= anArray;
            }
        }
        num1[0] = num_1;
        num2[0] = num_2;
    }

    private Boolean isBit0(int num, int index) {
        num = num >> index;
        return (num & 1) == 0;
    }

    /**
     * 题目描述
     * 输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。
     * 输出描述:
     *
     * @param array
     * @param sum
     * @return
     */
    public ArrayList<Integer> FindNumbersWithSum(int[] array, int sum) {
        ArrayList<Integer> list = new ArrayList<>();
        boolean flag = false;
        int start = 0;
        int end = array.length - 1;

        while (start < end) {
            int addSum = array[start] + array[end];
            if (addSum == sum) {
                flag = true;
                break;
            } else if (addSum < sum) {
                start++;
            } else {
                end--;
            }
        }

        if (flag) {
            if (array[start] < array[end]) {
                list.add(array[start]);
                list.add(array[end]);
            } else {
                list.add(array[end]);
                list.add(array[start]);
            }
        }
        return list;
    }

    /**
     * @param sum
     * @return
     */
    public ArrayList<ArrayList<Integer>> FindContinuousSequence(int sum) {
        ArrayList<ArrayList<Integer>> list = new ArrayList<>();
        if (sum <= 0) {
            return list;
        }

        int small = 1;
        int big = 2;
        int tmp = small + big;
        while (small < big && small <= sum / 2) {
            // int tmp = (small + big) * (big - small + 1) / 2;
            if (tmp > sum) {
                tmp -= small;
                small++;
            } else if (tmp == sum) {
                System.out.println("small:" + small + " big:" + big);
                ArrayList<Integer> res = new ArrayList<>();
                for (int i = small; i <= big; i++) {
                    res.add(i);
                }
                list.add(res);
                big++;
                tmp += big;
            } else {
                big++;
                tmp += big;
            }
        }

        return list;
    }

    /**
     * 题目描述
     * 汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。
     * 对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。
     * 例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！
     *
     * @param str
     * @param n
     * @return
     */
    public String LeftRotateString(String str, int n) {
        if (str.length() == 0) {
            return str;
        }

        char cs[] = str.toCharArray();
        reverseStr(cs, 0, n - 1);
        reverseStr(cs, n, str.length() - 1);
        reverseStr(cs, 0, str.length() - 1);
        return String.valueOf(cs);
    }

    public String ReverseSentence(String str) {
        char cs[] = str.toCharArray();
        int start = 0;
        int end = str.length() - 1;
        int flag = 0;
        for (int i = 0; i <= end; i++) {
            if (cs[i] == ' ') {
                reverseStr(cs, flag, i - 1);
                flag = i + 1;
            } else if (i == end) {
                reverseStr(cs, flag, i);
            }
        }

        reverseStr(cs, start, end);
        return String.valueOf(cs);
    }

    private void reverseStr(char[] chars, int start, int end) {
        while (start < end) {
            char c = chars[start];
            chars[start] = chars[end];
            chars[end] = c;
            start++;
            end--;
        }
    }

    /**
     * 判断数组是否连续
     *
     * @param numbers
     * @return
     */
    public boolean isContinuous(int[] numbers) {
        int nums[] = new int[14];
        int maxValue = -1;
        int minValue = -1;
        for (int tmp : numbers) {
            nums[tmp]++;
            if (tmp == 0) {
                continue;
            }

            if (nums[tmp] > 1) {
                return false;
            }

            if (minValue == -1) {
                minValue = tmp;
            }
            if (maxValue == -1) {
                maxValue = tmp;
            }

            if (minValue > tmp) {
                minValue = tmp;
            }
            if (maxValue < tmp) {
                maxValue = tmp;
            }
        }

        return maxValue - minValue <= numbers.length - 1;
    }

    /**
     * 题目描述
     * 每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。
     * HF作为牛客的资深元老,自然也准备了一些小游戏。
     * 其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。
     * 然后,他随机指定一个数m,让编号为0的小朋友开始报数。
     * 每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,
     * 并且不再回到圈中,从他的下一个小朋友开始,
     * 继续0...m-1报数....这样下去....直到剩下最后一个小朋友,
     * 可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。
     * 请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)
     * <p>
     * 如果没有小朋友，请返回-1
     * <p>
     * 约瑟夫环公式
     *
     * @param n
     * @param m
     * @return
     */
    public int LastRemaining_Solution(int n, int m) {
        //1. 递归
        // if (n == 0) {
        //     return 0;
        // }
        // if (n == 1) {
        //     return 1;
        // }
        //
        // return (LastRemaining_Solution(n - 1, m) + m) % n;

        // 2.模拟
        LinkedList<Integer> list = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            list.add(i);
        }

        int offset = 0;
        while (list.size() > 1) {
            offset = (offset + m - 1) % list.size();
            list.remove(offset);
        }

        return list.size() == 1 ? list.get(0) : -1;
    }

    public int Sum_Solution(int n) {
        int ans = n;
        boolean flag = ans > 0 && ((ans += Sum_Solution(n - 1)) > 0);
        return ans;
    }

    /**
     * 题目描述
     * 写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。
     *
     * <p>
     * 链接：https://www.nowcoder.com/questionTerminal/59ac416b4b944300b617d4f7f111b215?f=discussion
     * 来源：牛客网
     * <p>
     * 首先看十进制是如何做的： 5+7=12，三步走
     * 第一步：相加各位的值，不算进位，得到2。
     * 第二步：计算进位值，得到10. 如果这一步的进位值为0，那么第一步得到的值就是最终结果。
     * <p>
     * 第三步：重复上述两步，只是相加的值变成上述两步的得到的结果2和10，得到12。
     * <p>
     * 同样我们可以用三步走的方式计算二进制值相加： 5-101，7-111 第一步：相加各位的值，不算进位，得到010，二进制每位相加就相当于各位做异或操作，101^111。
     * <p>
     * 第二步：计算进位值，得到1010，相当于各位做与操作得到101，再向左移一位得到1010，(101&111)<<1。
     * <p>
     * 第三步重复上述两步， 各位相加 010^1010=1000，进位值为100=(010&1010)<<1。
     * 继续重复上述两步：1000^100 = 1100，进位值为0，跳出循环，1100为最终结果。
     *
     * @param num1
     * @param num2
     * @return
     */
    public int Add(int num1, int num2) {
        while (num2 != 0) {
            int temp = num1 ^ num2;
            num2 = (num1 & num2) << 1;
            num1 = temp;
        }
        return num1;
    }

    /**
     * 题目描述
     * 将一个字符串转换成一个整数，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0
     * 输入描述:
     * 输入一个字符串,包括数字字母符号,可以为空
     * 输出描述:
     * 如果是合法的数值表达则返回该数字，否则返回0
     *
     * @param str
     * @return
     */
    public int StrToInt(String str) {
        if (str == null || str.equals("") || str.equals("+") || str.equals("-")) {
            return 0;
        }

        boolean sign = true; //符号位
        int num = 0;
        for (int i = 0; i < str.length(); i++) {
            int m = str.charAt(i) - '0';
            if (m >= 0 && m <= 9) {
                if (sign && (num > Integer.MAX_VALUE / 10 ||
                        (num == Integer.MAX_VALUE / 10 && m > Integer.MAX_VALUE % 10))) {
                    return 0;
                }
                if (!sign && (num > Math.abs(Integer.MIN_VALUE / 10) ||
                        (num == Math.abs(Integer.MIN_VALUE / 10) && m > Math.abs(Integer.MIN_VALUE % 10)))) {
                    return 0;
                }
                num = m + num * 10;
            } else if ((str.charAt(i) == '-' || str.charAt(i) == '+') && i == 0) {
                sign = str.charAt(0) == '+';
            } else {
                return 0;
            }
        }

        return sign ? num : -num;
    }

    /**
     * 题目描述
     * 请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。
     * 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配
     *
     * @param str
     * @param pattern
     * @return
     */
    public boolean match(char[] str, char[] pattern) {
        if (str == null || pattern == null) {
            return false;
        }

        int strIndex = 0;
        int patternIndex = 0;
        return matchCore(str, strIndex, pattern, patternIndex);
    }

    private boolean matchCore(char[] str, int strIndex, char[] pattern, int patternIndex) {
        //越界判断
        if (strIndex == str.length && patternIndex == pattern.length) {
            return true;
        }

        if (strIndex != str.length && patternIndex == pattern.length) {
            return false;
        }

        //细节是魔鬼
        if (patternIndex + 1 < pattern.length && pattern[patternIndex + 1] == '*') {
            if (strIndex != str.length && (str[strIndex] == pattern[patternIndex] || pattern[patternIndex] == '.')) {
                return matchCore(str, strIndex, pattern, patternIndex + 2) ||
                        matchCore(str, strIndex + 1, pattern, patternIndex + 2) ||
                        matchCore(str, strIndex + 1, pattern, patternIndex);
            } else {
                return matchCore(str, strIndex, pattern, patternIndex + 2);
            }
        } else {
            if (strIndex != str.length && (str[strIndex] == pattern[patternIndex] || pattern[patternIndex] == '.')) {
                return matchCore(str, strIndex + 1, pattern, patternIndex + 1);
            }
        }

        return false;
    }

    //Insert one char from stringstream
    public void Insert(char ch) {
        if (stream[ch] == 0) {
            stream[ch] = curIndex++;
        } else {
            stream[ch] = -1;
        }
    }

    //return the first appearence once char in current stringstream
    public char FirstAppearingOnce() {
        int minIndex = -1;
        char ch = '#';
        for (int i = 0; i < stream.length; i++) {
            int index = stream[i];
            if (stream[i] > 0 && (index < minIndex || minIndex == -1)) {
                minIndex = index;
                ch = (char) i;
            }
        }

        return ch;
    }

    /**
     * 题目描述
     * 给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。
     *
     * @param pHead
     * @return
     */
    public ListNode EntryNodeOfLoop(ListNode pHead) {
        if (pHead == null) {
            return null;
        }
        //1.求是否有环
        ListNode quick = pHead;
        ListNode slow = pHead;
        while (true) {
            if (quick.next != null) {
                quick = quick.next.next;
                slow = slow.next;

                if (quick == null) {
                    return null;
                }
                if (slow == quick) {
                    break;
                }
            } else {
                return null;
            }
        }
        //2.求环的长度
        ListNode meetNode = quick;
        int n = 0;
        do {
            quick = quick.next;
            n++;
        } while (quick != meetNode);

        //3. 求相交结点
        quick = pHead;
        slow = pHead;
        while (n-- > 0) {
            quick = quick.next;
        }
        while (true) {
            if (quick == slow) {
                return quick;
            } else {
                quick = quick.next;
                slow = slow.next;
            }
        }
    }

    /**
     * 题目描述
     * 在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。
     * 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5
     *
     * @param pHead
     * @return
     */
    public ListNode deleteDuplication(ListNode pHead) {
        if (pHead == null) {
            return null;
        }

        ListNode fockHead = new ListNode(0);
        fockHead.next = pHead;
        ListNode pre = fockHead;
        ListNode after = fockHead.next;
        while (after != null) {
            boolean flag = false;
            while (after.next != null && after.val == after.next.val) {
                after = after.next;
                flag = true;
            }

            if (flag) {
                after = after.next;
            } else {
                pre.next = after;
                pre = pre.next;
                after = after.next;
            }
        }
        pre.next = null;
        return fockHead.next;
    }

    /**
     * 题目描述
     * 给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。
     *
     * @param pNode
     * @return
     */
    public TreeLinkNode GetNext(TreeLinkNode pNode) {
        if (pNode == null) {
            return null;
        }

        //右子树不为空，遍历求左子树
        if (pNode.right != null) {
            TreeLinkNode node = pNode.right;
            while (node.left != null) {
                node = node.left;
            }
            return node;
        }

        //右子树为空，看是否有父亲结点
        while (pNode.next != null) {
            //如果当前结点为父结点的左子树，那下一个结点就是此父结点
            if (pNode.next.left == pNode) {
                return pNode.next;
            } else {
                pNode = pNode.next;
            }
        }
        return null;
    }

    /**
     * 题目描述
     * 请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。
     *
     * @param pRoot
     * @return
     */
    public boolean isSymmetrical(TreeNode pRoot) {
        if (pRoot == null) {
            return true;
        }

        return isSymmetrical(pRoot.left, pRoot.right);
    }

    boolean isSymmetrical(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        if ((left == null && right != null) || (left != null && right == null)) {
            return false;
        }

        if (left.val == right.val) {
            return isSymmetrical(left.left, right.right) && isSymmetrical(left.right, right.left);

        }

        return false;
    }

    /**
     * 题目描述
     * 请实现两个函数，分别用来序列化和反序列化二叉树
     * <p>
     * 二叉树的序列化是指：把一棵二叉树按照某种遍历方式的结果以某种格式保存为字符串，从而使得内存中建立起来的二叉树可以持久保存。
     * 序列化可以基于先序、中序、后序、层序的二叉树遍历方式来进行修改，
     * 序列化的结果是一个字符串，序列化时通过 某种符号表示空节点（#），以 ！ 表示一个结点值的结束（value!）。
     * <p>
     * 二叉树的反序列化是指：根据某种遍历顺序得到的序列化字符串结果str，重构二叉树。
     *
     * @param root
     * @return
     */
    public String Serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        if (root == null) {
            sb.append("#!");
        } else {
            sb.append(root.val).append("!");
            sb.append(Serialize(root.left));
            sb.append(Serialize(root.right));
        }

        return sb.toString();
    }

    public TreeNode Deserialize(String str) {
        String[] strs = str.split("!");
        return Deserialize(strs);
    }

    private TreeNode Deserialize(String[] strs) {
        index++; //不管是否为null结点，都加1
        System.out.println(strs[index]);
        if (index > strs.length - 1) {
            return null;
        }

        if (strs[index].equals("#")) {
            return null;
        }

        TreeNode node = new TreeNode(Integer.parseInt(strs[index]));
        node.left = Deserialize(strs);
        node.right = Deserialize(strs);
        return node;
    }

    // 二叉搜索树的中序遍历是有序的。
    TreeNode KthNode(TreeNode pRoot, int k) {
        TreeNode node = null;
        if (pRoot != null) {
            node = KthNode(pRoot.left, k);
            if (node != null) {
                return node;
            }

            cur++;
            if (k == cur) {
                return pRoot;
            }
            node = KthNode(pRoot.right, k);
        }
        return node;
    }

    public void Insert(Integer num) {

        list.add(num);
        Collections.sort(list);
    }

    public Double GetMedian() {

        int len = list.size();

        double avg = 0;

        if (len % 2 == 1) {
            avg = list.get(len / 2);
        } else {
            avg = (list.get(len / 2) + list.get(len / 2 - 1)) / 2.0;
        }
        return avg;

    }

    /**
     * 题目描述
     * 给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，
     * 如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，
     * 他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}，
     * {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。
     *
     * @param num
     * @param size
     * @return
     */
    public ArrayList<Integer> maxInWindows(int[] num, int size) {
        ArrayList<Integer> res = new ArrayList<>();
        if (num == null || size == 0) {
            return res;
        }
        Deque<Integer> max = new LinkedList<>();
        for (int i = 0; i < num.length; i++) {
            while (!max.isEmpty() && num[max.peekLast()] < num[i]) {
                max.pollLast();
            }
            max.offerLast(i);
            if (max.peekFirst() <= i - size) {
                max.pollFirst();
            }
            if (i >= size - 1) {
                res.add(max.peekFirst());
            }
        }
        for (int i = 0; i < res.size(); i++) {
            res.set(i, num[res.get(i)]);
        }
        return res;
    }

    public boolean hasPath(char[] matrix, int rows, int cols, char[] str) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (findPath(new String(matrix), i, j, rows, cols, str, 0)) {
                    return true;
                }
            }

        }

        return false;

    }

    /**
     * 题目描述
     * 请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，
     * 每一步可以在矩阵中向左，向右，向上，向下移动一个格子。
     * 如果一条路径经过了矩阵中的某一个格子，则该路径不能再进入该格子。
     * 例如 a b c e s f c s a d e e 矩阵中包含一条字符串"bcced"的路径，
     * 但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。
     *
     * @param oid
     * @param x
     * @param y
     * @param rows
     * @param cols
     * @param str
     * @param start
     * @return
     */
    public boolean findPath(String oid, int x, int y,
                            int rows, int cols, char[] str, int start) {

        char matrix[] = oid.toCharArray();

        if (x < 0 || x >= rows || y < 0 || y >= cols) {
            return false;
        }

        if (matrix[x * cols + y] == str[start]) {
            matrix[x * cols + y] = '#';
            start++;
            if ((start) == str.length) {
                return true;
            }
        } else {
            return false;
        }

        return findPath(new String(matrix), x - 1, y, rows, cols, str, start)
                || findPath(new String(matrix), x + 1, y, rows, cols, str, start)
                || findPath(new String(matrix), x, y - 1, rows, cols, str, start)
                || findPath(new String(matrix), x, y + 1, rows, cols, str, start);
    }

    /**
     * 题目描述
     * 地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，
     * 但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，
     * 机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），
     * 因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？
     *
     * @param threshold
     * @param rows
     * @param cols
     * @return
     */
    public int movingCount(int threshold, int rows, int cols) {
        char arr[] = new char[rows * cols];

        return findPath(arr, threshold, 0, 0, rows, cols);
    }

    public int findPath(char arr[], int k, int x, int y, int rows,
                        int cols) {

        if (x < 0 || y < 0 || x >= cols || y >= rows || arr[x * rows + y] == '#') {
            return 0;
        }
        if ((getCount(x) + getCount(y)) > k) {
            return 0;
        }

        arr[x * rows + y] = '#';

        return
                findPath(arr, k, x - 1, y, rows, cols) +
                        findPath(arr, k, x + 1, y, rows, cols) +
                        findPath(arr, k, x, y - 1, rows, cols) +
                        findPath(arr, k, x, y + 1, rows, cols)
                        + 1;
    }

    public int getCount(int n) {

        int sum = 0;
        while (n != 0) {
            int c = n % 10;
            sum += c;
            n = n / 10;
        }

        return sum;
    }

    /**
     * 题目描述
     * 给你一根长度为n的绳子，请把绳子剪成整数长的m段（m、n都是整数，n>1并且m>1），每段绳子的长度记为k[0],k[1],...,k[m]。
     * 请问k[0]xk[1]x...xk[m]可能的最大乘积是多少？
     * 例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。
     * 输入描述:
     * 输入一个数n，意义见题面。（2 <= n <= 60）
     * 输出描述:
     * 输出答案。
     *
     * @param n
     * @return
     */
    public int cutRope(int n) {
        // n<=3的情况，m>1必须要分段，例如：3必须分成1、2；1、1、1 ，n=3最大分段乘积是2,
        if (n == 2) {
            return 1;
        }
        if (n == 3) {
            return 2;
        }
        int[] dp = new int[n + 1];
        /*
        下面3行是n>=4的情况，跟n<=3不同，4可以分很多段，比如分成1、3，
        这里的3可以不需要再分了，因为3分段最大才2，不分就是3。记录最大的。
         */
        dp[1] = 1;
        dp[2] = 2;
        dp[3] = 3;
        int res = 0;//记录最大的
        for (int i = 4; i <= n; i++) {
            for (int j = 1; j <= i / 2; j++) {
                res = Math.max(res, dp[j] * dp[i - j]);
            }
            dp[i] = res;
        }
        return dp[n];
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
    public static class ListNode {
        public int val;
        public ListNode next = null;

        public ListNode(int val) {
            this.val = val;
        }
    }

    public class RandomListNode {
        int label;
        RandomListNode next = null;
        RandomListNode random = null;

        RandomListNode(int label) {
            this.label = label;
        }
    }


}
