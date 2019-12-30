package singleton;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

/**
 * 单例的几种方式
 */
public abstract class Singleton {

    protected static Singleton mInstance;

    public static Singleton getInstance(String name) {
        if (mInstance == null) {
            mInstance = init(name);
        }
        return mInstance;
    }

    private static Singleton init(String name) {
        Singleton singleton = null;
        try {
            Class<?> clazz = Class.forName(name);
            Constructor<?> declaredConstructorBook = clazz.getDeclaredConstructor();
            declaredConstructorBook.setAccessible(true);
            Object objectBook = declaredConstructorBook.newInstance();
            singleton = (Singleton) objectBook;
        } catch (ClassNotFoundException | NoSuchMethodException | InstantiationException | IllegalAccessException | InvocationTargetException e) {
            e.printStackTrace();
        }
        return singleton;
    }

}
