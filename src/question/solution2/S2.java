package question.solution2;

/**
 * 二维数组中的查找
 * <p>
 * 题目描述
 * 在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
 * 请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
 * <p>
 * <p>
 * 例子： 查找4
 * 1 2 8 9
 * 2 4 9 12
 * 4 7 10 13
 * 6 8 11 15
 */
public class S2 {

    /**
     * 从右上角开始查找
     *
     * @param target 目标值
     * @param array  数组
     * @return 结果
     */
    public boolean FindFormRightTop(int target, int[][] array) {
        if (array == null || array.length == 0) {
            return false;
        }

        int rowSize = array.length; //行
        int colSize = array[0].length;

        int row = 0;
        int col = colSize - 1;
        while (row < rowSize && col >= 0) {
            int cur = array[row][col];
            if (cur == target) {
                return true;
            }
            if (cur > target) {
                col--;
            } else {
                row++;
            }
        }

        return false;
    }

    /**
     * 从左下角开始查找
     *
     * @param target 目标值
     * @param array  数组
     * @return 结果
     */
    public boolean FindFromLeftBottom(int target, int[][] array) {
        if (array == null || array.length == 0) {
            return false;
        }

        //注意细节，colSize和rowSize真正代表的意义
        int rowSize = array.length; //行
        int colSize = array[0].length;

        int row = rowSize - 1;
        int col = 0;
        while (row >= 0 && col < colSize) {
            int cur = array[row][col];
            if (cur == target) {
                return true;
            }
            if (cur > target) {
                row--;
            } else {
                col++;
            }
        }

        return false;
    }
}
