package question.solution2;

import java.util.ArrayList;

import question.solution1.Solution;

public class S4 {

    private ArrayList<Integer> res = new ArrayList<>();

    /**
     * 从尾到头打印链表
     * 题目描述
     * 输入一个链表，按链表从尾到头的顺序返回一个ArrayList。
     * <p>
     * 利用函数递归来模拟栈
     *
     * @param listNode 链表
     * @return 结果
     */
    public ArrayList<Integer> printListFromTailToHead(Solution.ListNode listNode) {
        if (listNode != null) {
            printListFromTailToHead(listNode.next);
            res.add(listNode.val);
        }
        return res;
    }

}
