import question.Solution;

/**
 * for java code
 * created by hutcwp
 * 2019.12.27
 */
public class Main {

    public static void main(String[] args) {
        Solution solution = new Solution();

        int pre[] = {1, 2, 4, 7, 3, 5, 6, 8};
        int in[] = {4, 7, 2, 1, 5, 3, 8, 6};
        Solution.TreeNode tree = solution.reConstructBinaryTree(pre, in);

        solution.printTree(tree);

    }
}
