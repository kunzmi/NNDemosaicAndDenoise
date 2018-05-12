namespace Demo
{
    partial class Form1
    {
        /// <summary>
        /// Erforderliche Designervariable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Verwendete Ressourcen bereinigen.
        /// </summary>
        /// <param name="disposing">True, wenn verwaltete Ressourcen gelöscht werden sollen; andernfalls False.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Vom Windows Form-Designer generierter Code

        /// <summary>
        /// Erforderliche Methode für die Designerunterstützung.
        /// Der Inhalt der Methode darf nicht mit dem Code-Editor geändert werden.
        /// </summary>
        private void InitializeComponent()
        {
            this.cmb_IsoValue = new System.Windows.Forms.ComboBox();
            this.btn_OpenImage = new System.Windows.Forms.Button();
            this.btn_Process = new System.Windows.Forms.Button();
            this.btn_Save = new System.Windows.Forms.Button();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.pictureBox2 = new System.Windows.Forms.PictureBox();
            this.pictureBox3 = new System.Windows.Forms.PictureBox();
            this.chk_Zoom = new System.Windows.Forms.CheckBox();
            this.btn_OpenPEF = new System.Windows.Forms.Button();
            this.Btn_ProcessPEF = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.lbl_ISO = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.btn_SaveNoisy = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox3)).BeginInit();
            this.SuspendLayout();
            // 
            // cmb_IsoValue
            // 
            this.cmb_IsoValue.FormattingEnabled = true;
            this.cmb_IsoValue.Items.AddRange(new object[] {
            "100",
            "200",
            "400",
            "800",
            "1600",
            "3200",
            "6400"});
            this.cmb_IsoValue.Location = new System.Drawing.Point(1107, 14);
            this.cmb_IsoValue.Name = "cmb_IsoValue";
            this.cmb_IsoValue.Size = new System.Drawing.Size(121, 21);
            this.cmb_IsoValue.TabIndex = 0;
            // 
            // btn_OpenImage
            // 
            this.btn_OpenImage.Location = new System.Drawing.Point(12, 12);
            this.btn_OpenImage.Name = "btn_OpenImage";
            this.btn_OpenImage.Size = new System.Drawing.Size(75, 23);
            this.btn_OpenImage.TabIndex = 1;
            this.btn_OpenImage.Text = "Open image";
            this.btn_OpenImage.UseVisualStyleBackColor = true;
            this.btn_OpenImage.Click += new System.EventHandler(this.btn_OpenImage_Click);
            // 
            // btn_Process
            // 
            this.btn_Process.Location = new System.Drawing.Point(93, 12);
            this.btn_Process.Name = "btn_Process";
            this.btn_Process.Size = new System.Drawing.Size(75, 23);
            this.btn_Process.TabIndex = 2;
            this.btn_Process.Text = "Process";
            this.btn_Process.UseVisualStyleBackColor = true;
            this.btn_Process.Click += new System.EventHandler(this.btn_Process_Click);
            // 
            // btn_Save
            // 
            this.btn_Save.Location = new System.Drawing.Point(1485, 12);
            this.btn_Save.Name = "btn_Save";
            this.btn_Save.Size = new System.Drawing.Size(75, 23);
            this.btn_Save.TabIndex = 3;
            this.btn_Save.Text = "Save result";
            this.btn_Save.UseVisualStyleBackColor = true;
            this.btn_Save.Click += new System.EventHandler(this.btn_Save_Click);
            // 
            // pictureBox1
            // 
            this.pictureBox1.Location = new System.Drawing.Point(12, 41);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(512, 512);
            this.pictureBox1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pictureBox1.TabIndex = 4;
            this.pictureBox1.TabStop = false;
            // 
            // pictureBox2
            // 
            this.pictureBox2.Location = new System.Drawing.Point(530, 41);
            this.pictureBox2.Name = "pictureBox2";
            this.pictureBox2.Size = new System.Drawing.Size(512, 512);
            this.pictureBox2.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pictureBox2.TabIndex = 4;
            this.pictureBox2.TabStop = false;
            // 
            // pictureBox3
            // 
            this.pictureBox3.Location = new System.Drawing.Point(1048, 41);
            this.pictureBox3.Name = "pictureBox3";
            this.pictureBox3.Size = new System.Drawing.Size(512, 512);
            this.pictureBox3.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pictureBox3.TabIndex = 4;
            this.pictureBox3.TabStop = false;
            // 
            // chk_Zoom
            // 
            this.chk_Zoom.AutoSize = true;
            this.chk_Zoom.Location = new System.Drawing.Point(1048, 18);
            this.chk_Zoom.Name = "chk_Zoom";
            this.chk_Zoom.Size = new System.Drawing.Size(53, 17);
            this.chk_Zoom.TabIndex = 5;
            this.chk_Zoom.Text = "Zoom";
            this.chk_Zoom.UseVisualStyleBackColor = true;
            this.chk_Zoom.CheckedChanged += new System.EventHandler(this.chk_Zoom_CheckedChanged);
            // 
            // btn_OpenPEF
            // 
            this.btn_OpenPEF.Location = new System.Drawing.Point(530, 12);
            this.btn_OpenPEF.Name = "btn_OpenPEF";
            this.btn_OpenPEF.Size = new System.Drawing.Size(75, 23);
            this.btn_OpenPEF.TabIndex = 6;
            this.btn_OpenPEF.Text = "Open PEF";
            this.btn_OpenPEF.UseVisualStyleBackColor = true;
            this.btn_OpenPEF.Click += new System.EventHandler(this.btn_OpenPEF_Click);
            // 
            // Btn_ProcessPEF
            // 
            this.Btn_ProcessPEF.Location = new System.Drawing.Point(611, 12);
            this.Btn_ProcessPEF.Name = "Btn_ProcessPEF";
            this.Btn_ProcessPEF.Size = new System.Drawing.Size(90, 23);
            this.Btn_ProcessPEF.TabIndex = 7;
            this.Btn_ProcessPEF.Text = "Process PEF";
            this.Btn_ProcessPEF.UseVisualStyleBackColor = true;
            this.Btn_ProcessPEF.Click += new System.EventHandler(this.Btn_ProcessPEF_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(707, 19);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(47, 13);
            this.label1.TabIndex = 8;
            this.label1.Text = "File ISO:";
            // 
            // lbl_ISO
            // 
            this.lbl_ISO.AutoSize = true;
            this.lbl_ISO.Location = new System.Drawing.Point(760, 19);
            this.lbl_ISO.Name = "lbl_ISO";
            this.lbl_ISO.Size = new System.Drawing.Size(0, 13);
            this.lbl_ISO.TabIndex = 9;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(230, 555);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(77, 13);
            this.label2.TabIndex = 10;
            this.label2.Text = "Original Bitmap";
            this.label2.TextAlign = System.Drawing.ContentAlignment.TopCenter;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(678, 555);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(217, 13);
            this.label3.TabIndex = 10;
            this.label3.Text = "Noisy and simple debayering (=input to CNN)";
            this.label3.TextAlign = System.Drawing.ContentAlignment.TopCenter;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(1257, 555);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(95, 13);
            this.label4.TabIndex = 10;
            this.label4.Text = "Final result of CNN";
            this.label4.TextAlign = System.Drawing.ContentAlignment.TopCenter;
            // 
            // btn_SaveNoisy
            // 
            this.btn_SaveNoisy.Location = new System.Drawing.Point(967, 14);
            this.btn_SaveNoisy.Name = "btn_SaveNoisy";
            this.btn_SaveNoisy.Size = new System.Drawing.Size(75, 23);
            this.btn_SaveNoisy.TabIndex = 3;
            this.btn_SaveNoisy.Text = "Save noisy";
            this.btn_SaveNoisy.UseVisualStyleBackColor = true;
            this.btn_SaveNoisy.Click += new System.EventHandler(this.btn_SaveNoisy_Click);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1567, 579);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.lbl_ISO);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.Btn_ProcessPEF);
            this.Controls.Add(this.btn_OpenPEF);
            this.Controls.Add(this.chk_Zoom);
            this.Controls.Add(this.pictureBox3);
            this.Controls.Add(this.pictureBox2);
            this.Controls.Add(this.pictureBox1);
            this.Controls.Add(this.btn_SaveNoisy);
            this.Controls.Add(this.btn_Save);
            this.Controls.Add(this.btn_Process);
            this.Controls.Add(this.btn_OpenImage);
            this.Controls.Add(this.cmb_IsoValue);
            this.Name = "Form1";
            this.Text = "Form1";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox3)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.ComboBox cmb_IsoValue;
        private System.Windows.Forms.Button btn_OpenImage;
        private System.Windows.Forms.Button btn_Process;
        private System.Windows.Forms.Button btn_Save;
        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.PictureBox pictureBox2;
        private System.Windows.Forms.PictureBox pictureBox3;
        private System.Windows.Forms.CheckBox chk_Zoom;
        private System.Windows.Forms.Button btn_OpenPEF;
        private System.Windows.Forms.Button Btn_ProcessPEF;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label lbl_ISO;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Button btn_SaveNoisy;
    }
}

