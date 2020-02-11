package question.solution2;

public class S3 {

    /**
     * 题目描述
     * 请实现一个函数，将一个字符串中的每个空格替换成“%20”。
     * 例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
     *
     * @param str 原始字符串
     * @return
     */
    public String replaceSpace(StringBuffer str) {
        if (str == null || str.length() == 0) {
            return str == null ? null : str.toString();
        }

        int n = 0;
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == ' ') {
                n++;
            }
        }

        char cs[] = new char[str.length() + n * 2];
        int pre = str.length() - 1;
        int end = cs.length - 1;
        for (int i = pre; i >= 0; i--) {
            if (str.charAt(i) != ' ') {
                cs[end--] = str.charAt(i);
            } else {
                cs[end--] = '0';
                cs[end--] = '2';
                cs[end--] = '%';
            }
        }
        return String.valueOf(cs);
    }
}
