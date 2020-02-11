package question.solution2;

public class S7 {

    public static void main(String[] args) {
        S7 s7 = new S7();
        System.out.println(s7.minNumberInRotateArray(new int[]{1, 2, 3, 4, 5}));
    }

    /**
     * 旋转数组的最小值
     * 题目描述
     * 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
     * 输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
     * 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
     * NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
     *
     * @param array 数组
     * @return 最小值
     * <p>
     * 1，2，2，2，特殊  -》 3，2，2，2 ,2 ,2
     * <p>
     * 2，2，2，2，特殊  -》
     * <p>
     * 1 1  0 1 1
     * <p>
     * 1，2，3，4，5 正常 -》 3 ,4 ,5 ,1 ,2
     */
    public int minNumberInRotateArray(int[] array) {
        if (array == null || array.length <= 0) {
            return 0;
        }

        int start = 0;
        int end = array.length - 1;
        int mid = (start + end) / 2;
        if (array[end] == array[mid] && array[start] == array[mid]) {
            //遍历
            int min = array[0];
            for (int i = 1; i < array.length; i++) {
                if (min > array[i]) {
                    min = array[i];
                }
            }
            return min;
        } else {
            while (array[start] < array[end]) {
                mid = (start + end) / 2;
                if (array[mid] >= array[start] || array[mid] >= array[end]) {
                    start = mid;
                } else {
                    end = mid;
                }
                if (start >= end - 1) {
                    break;
                }
            }
            return array[end];
        }
    }
}
