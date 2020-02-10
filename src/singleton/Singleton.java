package singleton;

/**
 * 单例的几种方式
 */
public class Singleton {

    private static Singleton mInstance = null;

    private Singleton() {
        //构造函数私有化
    }

    public static Singleton getInstance() {
        if (mInstance == null) {
            mInstance = new Singleton();
        }
        return mInstance;
    }
}
