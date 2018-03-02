package com.hipad.featuresapp;

import android.annotation.TargetApi;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.ImageView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static android.Manifest.permission.READ_EXTERNAL_STORAGE;
import static android.os.Build.VERSION_CODES.M;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("opencv_java");
        System.loadLibrary("native-lib");
    }
    private BaseLoaderCallback mOpenCVCallBack=new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            Log.e("www", "OpenCV loadeding callback");
            switch(status){
                 case LoaderCallbackInterface.SUCCESS:
                     // TODO: 2018/2/27 完成我们自己的操作
                     Log.e("www", "OpenCV loaded successfully");
                 break;
                 default:
                     Log.e("www", "OpenCV loaded failed");
                     super.onManagerConnected(status);
                 break;
            }
        }
    };
    private Bitmap originalBitmap;
    private Mat originalMat;
    private Bitmap currentBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //Initializing OpenCV Package Manager
//        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_10,this,mOpenCVCallBack);
        setContentView(R.layout.activity_main);
        if(hasPermissions()){

        } else if(Build.VERSION.SDK_INT >= M){
            requestPermissions();
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.main_menu,menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
            int id=item.getItemId();
            if (id==R.id.action_settings){
                return true;
            }else if(id==R.id.open_gallery){
                Intent intent=new Intent(Intent.ACTION_PICK, Uri.parse("content://media/internal/images/media"));
                startActivityForResult(intent,0);
            }else if(id==R.id.DoG){//高斯差分 different of gaussian
                differenceOfGaussian();
            }else if(id==R.id.CannyEdges){//Canny边缘检测器 被认为是最优方法
                canny();
            }else if(id==R.id.SobelFilter){//sobel算子边缘检测实现
                sobel();
            }else if(id==R.id.HarrisCorners){//harris角点检测
                harrisCorner();
            }else if(id==R.id.HoughLines){//霍夫变换之霍夫直线
                houghLines();
            }else if(id==R.id.HoughCircles){//霍夫变换之霍夫圆
                houghCircles();
            }else if(id==R.id.Contours){//轮廓检测
                contours();
            }
        return super.onOptionsItemSelected(item);
    }

    /**
     * 轮廓检测
     */
    private void contours() {
        Mat grayMat=new Mat();
        Mat cannyEdges=new Mat();
        Mat hierarchy=new Mat();
        //保存所有轮廓列表
        List<MatOfPoint> contourList=new ArrayList<>();
        Imgproc.cvtColor(originalMat,grayMat,Imgproc.COLOR_BGR2GRAY);
        Imgproc.Canny(grayMat,cannyEdges,10,100);
        //找出轮廓
        Imgproc.findContours(cannyEdges,contourList,hierarchy,Imgproc.RETR_LIST,Imgproc.CHAIN_APPROX_SIMPLE);
        //在心的图像上绘制轮廓
        Mat contous=new Mat();
        contous.create(cannyEdges.rows(),cannyEdges.cols(),CvType.CV_8UC3);
        Random r=new Random();
        for (int i=0;i<contourList.size();i++){
            Imgproc.drawContours(contous,contourList,i,new Scalar(r.nextInt(255),r.nextInt(255),r.nextInt(255),1));
        }
        Utils.matToBitmap(contous,currentBitmap);
        loadImageToImageView();
    }

    /**
     * 霍夫圆
     */
    private void houghCircles() {
        Mat grayMat=new Mat();
        Mat cannyEdges=new Mat();
        Mat circles=new Mat();
        Imgproc.cvtColor(originalMat,grayMat,Imgproc.COLOR_BGR2GRAY);
        Imgproc.Canny(grayMat,cannyEdges,10,100);
        Imgproc.HoughCircles(cannyEdges,circles,Imgproc.CV_HOUGH_GRADIENT,1,cannyEdges.rows()/15);
        //或者最后的参数给成grayMat.rows()/8
        Mat houghCircles=new Mat();
        houghCircles.create(cannyEdges.rows(),cannyEdges.cols(),CvType.CV_8UC1);
        //在图像上绘制圆形
        for (int i=0;i<circles.cols();i++){
            double[] parameters = circles.get(0, i);
            double x,y;
            int r;
            x=parameters[0];
            y=parameters[1];
            r= (int) parameters[2];
            Point center=new Point(x,y);
            Core.circle(houghCircles,center,r,new Scalar(255,0,0),1);
        }
        Utils.matToBitmap(houghCircles,currentBitmap);
        loadImageToImageView();
    }

    /**
     * 霍夫直线
     */
    private void houghLines() {
        Mat grayMat=new Mat();
        //选择一种边缘检测，检测边缘 我这里采用canny算子检测算法
        Mat cannyEdges=new Mat();
        Mat lines=new Mat();
        //转成灰度图像
        Imgproc.cvtColor(originalMat,grayMat,Imgproc.COLOR_BGR2GRAY);
        //canny 边缘检测
        Imgproc.Canny(grayMat,cannyEdges,10,100);
        //canny的输出作为该方法的输入，参数 3，4指定像素中r和寺塔解析度，5，6参数是一条直线上点数的阈值，少于该值的直线被舍弃
        Imgproc.HoughLinesP(cannyEdges,lines,1,Math.PI/180,50,20,20);
        //新建Mat结构图像用来把直线画在该Mat上
        Mat houghLines=new Mat();
        houghLines.create(cannyEdges.rows(),cannyEdges.cols(),CvType.CV_8UC1);
        //在图像上绘制直线
        for (int i=0;i<lines.cols();i++){
            double[] points = lines.get(0, i);
            double x1,y1,x2,y2;
            x1=points[0];
            y1=points[1];
            x2=points[2];
            y2=points[3];
            Point pt1=new Point(x1,y1);
            Point pt2=new Point(x2,y2);
            //在一幅图像上绘制直线
            Core.line(houghLines,pt1,pt2,new Scalar(255,0,0),1);
        }
        Utils.matToBitmap(houghLines,currentBitmap);
        loadImageToImageView();
    }

    /**
     * 角点检测
     */
    private void harrisCorner() {
        Mat grayMat=new Mat();
        Mat conners=new Mat();
        //将图像转化成灰度图像
        Imgproc.cvtColor(originalMat,grayMat,Imgproc.COLOR_BGR2GRAY);
        //找出角点
        Mat tempDst=new Mat();
        Imgproc.cornerHarris(grayMat,tempDst,2,3,0.04);

        //归一化Harris角点的输出
        Mat tempDitNorm=new Mat();
        Core.normalize(tempDst,tempDitNorm,0,255,Core.NORM_MINMAX);
        Core.convertScaleAbs(tempDitNorm,conners);
        //在新的图像上绘制角点
        Random r=new Random();
        for (int i=0;i<tempDitNorm.cols();i++){
            for (int j=0;j<tempDitNorm.rows();j++){
                double[] value = tempDitNorm.get(j, i);
                if (value[0] > 150)
                    Core.circle(conners, new Point(i, j), 5, new Scalar(r.nextInt(255)), 2);
            }
            }

            //将mat转换回位图
        Utils.matToBitmap(conners,currentBitmap);
        loadImageToImageView();
        }
    /**
     * sobel算子计算图像边缘
     */
    private void sobel() {
        Mat grayMat=new Mat();
        Mat sobel=new Mat();
        //分别用于保存梯度和绝对梯度的Mat
        Mat grad_x=new Mat();
        Mat abs_grad_x=new Mat();
        Mat grad_y=new Mat();
        Mat abs_grad_y=new Mat();

        //转换为灰度图
        Imgproc.cvtColor(originalMat,grayMat,Imgproc.COLOR_BGR2GRAY);
        //计算水平方向的梯度
        Imgproc.Sobel(grayMat,grad_x, CvType.CV_16S,1,0,3,1,0);
        //计算垂直方向的梯度
        Imgproc.Sobel(grayMat,grad_y,CvType.CV_16S,0,1,3,1,0);
        //计算两个方向上的梯度绝对值
        Core.convertScaleAbs(grad_x,abs_grad_x);
        Core.convertScaleAbs(grad_y,abs_grad_y);
        //计算结果梯度
        Core.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,1,sobel);
        //将Mat转换成位图
        Utils.matToBitmap(sobel,currentBitmap);
        loadImageToImageView();
    }

    /***
     * canny检测算子边缘检测实现 步骤：
     * 1。平滑图像即消除噪声，采用高斯平滑滤波卷积降噪
     * 2。计算图像的梯度 并将梯度分类为垂直、水平和斜对角线
     * 3。非最大值抑制，利用计算得到的梯度方向检查某一像素在梯度局部最大，最大择抑制
     * 4。用滞后阈值化选择边缘
     */
    private void canny() {
        Mat grayMat=new Mat();
        Mat cannyEdges=new Mat();
        //将图像转换成灰度图像
        Imgproc.cvtColor(originalMat,grayMat,Imgproc.COLOR_BGR2GRAY);

        Imgproc.Canny(grayMat,cannyEdges,10,100);
        //将mat转换回bitmap位图
        Utils.matToBitmap(cannyEdges,currentBitmap);
        loadImageToImageView();
    }

    /**
     * 1,将给定的图像转换为灰度图像
     * 2，用两个不同的模糊半径对灰度图像执行高斯模糊
     * 3，将产生的两幅图像相减，得到一幅只包含边缘点结果的图像。
     */
    private void differenceOfGaussian() {
        // TODO: 2018/3/1
        Mat grayMat=new Mat();
        Mat blur1=new Mat();
        Mat blur2=new Mat();
        //将图像转化成灰度图像
        //遇到参数Cn代表的是通道数 如果没有默认会把原图的通道数作为结果图像的通道数
        Imgproc.cvtColor(originalMat,grayMat,Imgproc.COLOR_BGR2GRAY);
        //以两个不同模糊半径对图像进行模糊处理
        Imgproc.GaussianBlur(grayMat,blur1,new Size(15,15),5);
        Imgproc.GaussianBlur(grayMat,blur2,new Size(21,21),5);

        //将两幅模糊后的图像相减
        Mat DoG=new Mat();
        Core.absdiff(blur1,blur2,DoG);

        //反转二值阈值化
        Core.multiply(DoG,new Scalar(100), DoG);
        Imgproc.threshold(DoG,DoG,50,255,Imgproc.THRESH_BINARY_INV);
        //将mat转换回位图
        Utils.matToBitmap(DoG,currentBitmap);
        loadImageToImageView();
    }

    /**
     * 申请权限
     */
    private static final int REQUEST_PERMISSIONS = 1;
    @TargetApi(M)
    private void requestPermissions() {
        if (!shouldShowRequestPermissionRationale(READ_EXTERNAL_STORAGE)) {
            requestPermissions(new String[]{READ_EXTERNAL_STORAGE}, REQUEST_PERMISSIONS);
            return;
        }
        new AlertDialog.Builder(this)
                .setMessage("giving the read storage permission")
                .setCancelable(false)
                .setPositiveButton(android.R.string.ok, (dialog, which) ->
                        requestPermissions(new String[]{READ_EXTERNAL_STORAGE}, REQUEST_PERMISSIONS))
                .setNegativeButton(android.R.string.cancel, null)
                .create()
                .show();
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == REQUEST_PERMISSIONS) {
            // we request 2 permissions
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {

            }
        }
    }
    private boolean hasPermissions() {
        PackageManager pm = getPackageManager();
        String packageName = getPackageName();
        int granted =pm.checkPermission(READ_EXTERNAL_STORAGE, packageName);
        return granted == PackageManager.PERMISSION_GRANTED;
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode==0&&resultCode==RESULT_OK&&data!=null){
            Uri selectedImage = data.getData();
            String[] filePathColum={MediaStore.Images.Media.DATA};
            //length=1 就是我们选择的那个图片
            Log.e("www", ""+filePathColum.length);
            Log.e("www",filePathColum.toString());
            Cursor cursor = getContentResolver().query(selectedImage, filePathColum, null, null, null);
            cursor.moveToFirst();
            int columnIndex = cursor.getColumnIndex(filePathColum[0]);
            //拿到的media数据库的列名 _data
            Log.e("www", filePathColum[0]);
            String picturePath = cursor.getString(columnIndex);
            //图像的决定路径/storage/emulated/0/image.jpg
            Log.e("www", picturePath);

            cursor.close();

            //加速图像的载入，对图像进行压缩
            BitmapFactory.Options options=new BitmapFactory.Options();
            options.inSampleSize=2;
            Bitmap temp = BitmapFactory.decodeFile(picturePath, options);
            //获取图像的方向信息
            int orientation=0;
            try {
                ExifInterface imgParams=new ExifInterface(picturePath);
                orientation = imgParams.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED);

            } catch (IOException e) {
                e.printStackTrace();
            }
            //旋转图像，得到正确的方向
            Matrix rotate90=new Matrix();
            rotate90.postRotate(orientation);
            originalBitmap = rotateBitmap(temp, orientation);
            if (null==originalBitmap){
                Log.e("www","null obj");
                return;
            }
            //将位图转换位Mat
            Bitmap tempBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
            originalMat=new Mat(tempBitmap.getHeight(),tempBitmap.getWidth(), CvType.CV_8U);
            Utils.bitmapToMat(tempBitmap,originalMat);
            currentBitmap=originalBitmap.copy(Bitmap.Config.ARGB_8888,false);
            loadImageToImageView();

        }
    }

    /**
     * 把image显示到imageview上面
     */
    private void loadImageToImageView() {
        ImageView imgView = (ImageView) findViewById(R.id.image_view);
        imgView.setImageBitmap(currentBitmap);
    }
    //Function to rotate bitmap according to image parameters
    public static Bitmap rotateBitmap(Bitmap bitmap, int orientation) {
        Matrix matrix = new Matrix();
        switch (orientation) {
            case ExifInterface.ORIENTATION_NORMAL:
                return bitmap;
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL:
                matrix.setScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.setRotate(180);
                break;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL:
                matrix.setRotate(180);
                matrix.postScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_TRANSPOSE:
                matrix.setRotate(90);
                matrix.postScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.setRotate(90);
                break;
            case ExifInterface.ORIENTATION_TRANSVERSE:
                matrix.setRotate(-90);
                matrix.postScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_ROTATE_270:
                matrix.setRotate(-90);
                break;
            default:
                return bitmap;
        }
        try {
            Bitmap bmRotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
            bitmap.recycle();
            return bmRotated;
        } catch (OutOfMemoryError e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
}
