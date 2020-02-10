package question.solution1;

import java.util.Stack;

/**
 * Solution2
 */
public class Solution2 {

    Stack<Integer> stack = new Stack<>();
    Stack<Integer> stackHelper = new Stack<>();

    public void push(int node) {
        stack.push(node);
        if (stackHelper.isEmpty()) {
            stackHelper.push(node);
        } else {
            int top = stackHelper.peek();
            stackHelper.push(Math.min(top, node));
        }
    }

    public void pop() {
        stack.pop();
        stackHelper.pop();

    }

    public int top() {
        return stack.peek();
    }

    public int min() {
        return stackHelper.peek();
    }

}
