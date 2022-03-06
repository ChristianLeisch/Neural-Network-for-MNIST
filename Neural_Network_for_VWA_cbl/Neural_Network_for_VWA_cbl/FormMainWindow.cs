using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;

namespace NeuralNetwork
{
    public partial class Form_main_window : Form
    {
        public Form_main_window()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        const int numberpics = 60000;
        

        const int inputnodes = 784;
        const int hiddennodes = 100;
        const int outputnodes = 10;
        const double learningrate = 0.2;

        double[,] wih = new double[hiddennodes, inputnodes];
        double[,] who = new double[outputnodes, hiddennodes];

        double[] hiddenin = new double[hiddennodes];
        double[] hiddenout = new double[hiddennodes];

        double[] lastin = new double[outputnodes];
        double[] lastout = new double[outputnodes];

        double[,] images = new double[numberpics, 785];

        Random random = new Random();

        private void btnInit_Click(object sender, EventArgs e)
        {
            string Path = @"E:\VWA\datasets\mnist_test.csv";
            string Path1 = @""+textBox7.Text;
            var reader = new StreamReader(File.OpenRead(Path1));
            for(int i =0; i<numberpics;i++)
            {
                var line = reader.ReadLine();
                var values = line.Split(',');
                int[] vals = Array.ConvertAll(values, int.Parse);
                images[i, 0] = vals[0];
                for (int j = 1; j < 785; j++)
                {
                    images[i, j] = (Convert.ToDouble(vals[j])/255*0.99)+0.01;
                }
            }
            
            for(int i =0;i<inputnodes; i++)
            {
                for (int j = 0; j < hiddennodes; j++)
                {
                    wih[j, i] = (Convert.ToDouble(random.Next(0, 20)) - 10) / 10;
                }
            }

            for(int i = 0; i < hiddennodes; i++)
            {
                for (int j = 0; j < outputnodes; j++)
                {
                    who[j, i] = (Convert.ToDouble(random.Next(0, 20)) - 10) / 10;
                }
            }
            lbl1.Text = "init done";
        }

        private void calculate(int n)
        {
            for(int i =0; i<hiddennodes;i++)
            {
                hiddenin[i] = 0;
                for (int j = 0; j < inputnodes; j++)
                {
                    hiddenin[i] += images[n, j + 1] * wih[i,j];
                }
                hiddenout[i] = sigmoid(hiddenin[i]);
            }

            for (int i = 0; i < outputnodes; i++)
            {
                lastin[i] = 0;
                for (int j = 0; j < hiddennodes; j++)
                {
                    lastin[i] += hiddenout[j] * who[i, j];
                }
                lastout[i] = sigmoid(lastin[i]);
             }
        }

        private double sigmoid(double sum)
        {
            return 1 / (1 + Math.Pow(Math.E, (-sum)));
        }

        private void train(int n)
        {
            calculate(n);
            double[] oerrors = new double[outputnodes];
            double[] herrors = new double[hiddennodes];

            for(int i =0;i<outputnodes;i++)
            {
                if((i)==images[n,0])
                {
                    oerrors[i] = 0.99 - lastout[i];
                }
                else
                {
                    oerrors[i] = 0.01 - lastout[i];
                }
            }

            for(int i=0;i<hiddennodes;i++)
            {
                for(int j =0;j<outputnodes;j++)
                {
                    herrors[i] += oerrors[j] * who[j, i];
                }
            }

            for(int j =0;j<outputnodes;j++)
            {
                for(int i =0;i<hiddennodes;i++)
                {
                    who[j, i] += learningrate * oerrors[j] * lastout[j] * (1 - lastout[j]) * hiddenout[i];
                }
            }
            
            for (int j = 0; j < hiddennodes; j++)
            {
                for (int i = 0; i < inputnodes; i++)
                {
                    wih[j, i] += learningrate * herrors[j] * hiddenout[j] * (1 - hiddenout[j]) * images[n,i+1]; //nochmal checken, Seite 86
                }
            }
            
            
        }

        private void btnTrain_Click(object sender, EventArgs e)
        {
            int o = Int32.Parse(textBox1.Text);
            int n = Int32.Parse(textBox2.Text);
            int p = Int32.Parse(textBox3.Text);
            int c = (n - o) * p;
            int r = 0;
            for (int j = 0; j < p; j++)
            {
                for (int i = o; i < n; i++)
                {

                    train(i);
                    r++;
                    showPercentage(r, c);
                }
            }
        }

        private void showPercentage(int i, int a)
        {
            lbltest.Text = i + "/" + a;
            lbltest.Invalidate();
            lbltest.Update();
            lbltest.Refresh();
            Application.DoEvents();
        }

        private void btnTest_Click(object sender, EventArgs e)
        {
            int o = Int32.Parse(textBox4.Text);
            int n = Int32.Parse(textBox5.Text);
            double correct = 0;
            for (int i = o; i < n; i++)
            {
                correct = correct + test(i);
            }
            lblacc.Text = "accuracy:" + correct / (n - o) * 100 +"%";
        }

        private int test(int n)
        {
            int realnumber = Convert.ToInt32(images[n, 0]);
            calculate(n);
            int max = 0;
            for (int i = 0; i < 10; i++)
            {
                if (lastout[i] > lastout[max])
                {
                    max = i;
                }
            }
            if (max == realnumber)
            {
                return 1;
            }
            else
            {
                return 0;
            }

        }
        private int predict(int n)
        {

            calculate(n);
            int max = 0;
            for (int i = 0; i < 10; i++)
            {
                if (lastout[i] > lastout[max])
                {
                    max = i;
                }
            }
            return max;
        }

        private void btnimage_Click(object sender, EventArgs e)
        {
            int m = Int32.Parse(textBox6.Text);
            int scale = 8;
            Bitmap bitimage = new Bitmap(28, 28);
            for (int Xcount = 0; Xcount < 28; Xcount++)
            {
                for (int Ycount = 0; Ycount < 28; Ycount++)
                {
                    int address = Ycount * 28 + Xcount + 1;
                    int color = Convert.ToInt32((images[m, address]-0.01)*255/0.99);
                    bitimage.SetPixel(Xcount, Ycount, Color.FromArgb(color, color, color));
                }
            }

            Bitmap scaledbitimage = new Bitmap(28 * scale, 28 * scale);
            for (int Xcount = 0; Xcount < 28; Xcount++)
            {
                for (int Ycount = 0; Ycount < 28; Ycount++)
                {
                    Color color = bitimage.GetPixel(Xcount, Ycount);
                    for (int k = 0; k < 8; k++)
                    {
                        for (int l = 0; l < 8; l++)
                        {
                            int x = Xcount * 8 + k;
                            int y = Ycount * 8 + l;
                            scaledbitimage.SetPixel(x, y, color);
                        }
                    }
                }

            }
            pictureBox1.Image = scaledbitimage;
            lblout.Text = null;
            calculate(m);
            for (int i = 0; i < 10; i++)
            {
                lblout.Text = lblout.Text + i + ":" + lastout[i] + "\n";
            }
            lblout.Text = lblout.Text + "predicted: " + predict(m) + ", real:" + images[m, 0];
        }

        private void textBox7_TextChanged(object sender, EventArgs e)
        {

        }

        private void lbl1_Click(object sender, EventArgs e)
        {

        }
    }
}
