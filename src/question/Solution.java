package question;

public class Solution {

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
}
