package question.solution2;

/**
 * 第二遍
 */
public class S1 {

    /**
     * 题目: 数组中重复的数字
     * 题目描述
     * 在一个长度为n的数组里的所有数字都在0到n-1的范围内。
     * 数组中某些数字是重复的，但不知道有几个数字是重复的。
     * 也不知道每个数字重复几次。
     * 请找出数组中任意一个重复的数字。
     * 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。
     * <p>
     * <p>
     * 思路：
     * 1. 排序后，遍历一边找出，时间复杂度O(log(n))
     * 2. 用哈希表，遍历一边找出，时间复杂度O{O(1)},空间复杂度O(n)
     * 3. 因为数组元素为0~n,那将每一位与i进行对比，如果相等就往下走。
     * 不相等，判断nums[i]是否与nums[nums[i]]相等，相等则找到并返回退出，不相等则交换并重新上面步骤
     *
     * @param numbers     原数组
     * @param length      数组长度
     * @param duplication 结果返回数组
     * @return 是否包含
     */
    public boolean duplicate(int numbers[], int length, int[] duplication) {
        if (numbers == null || length == 0) {
            return false;
        }

        for (int i = 0; i < length; i++) {
            while (numbers[i] != i) {
                if (numbers[i] >= length) {
                    return false;
                }

                if (numbers[numbers[i]] == numbers[i]) {
                    duplication[0] = numbers[i];
                    return true;
                } else {
                    swap(numbers, i, numbers[i]);
                }
            }

        }

        return false;
    }

    private void swap(int nums[], int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
}
