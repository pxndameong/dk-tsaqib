import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras import backend as K
import numpy as np
from scipy import spatial
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import os
from shapely.geometry import box
from mpl_toolkits.axes_grid1 import make_axes_locatable

#ISI CUSTOM
PilihTahun = 2013  # Pilih tahun yang diinginkan

for PilihBulan in range(7 , 8):
    print(f"\nProcessing Month {PilihBulan} Year {PilihTahun}")
    
    maenya=0.8

    persentotaldata= 1
    angkapersendata = persentotaldata*100
    templatesample = f"51var_{angkapersendata}%_stasiun"

    folder_save_hasil = "C:\FILE\PROYEK PROF EDDY\DATA\DK_MERGE_TSAQIB/1 YEAR/"
    output_dir_png = folder_save_hasil

    kolom_terpilih = [
    "lat", "lon", "cape", "hcc", "mcc", "msl", "slhf", "sshf", "t2m", "tclw", "tcwv",
    "q_1000", "q_925", "q_850", "q_600", "q_500", "q_200",
    "r_1000", "r_925", "r_850", "r_600", "r_500", "r_200",
    "t_1000", "t_925", "t_850", "t_600", "t_500", "t_200",
    "u_1000", "u_925", "u_850", "u_600", "u_500", "u_200",
    "v_1000", "v_925", "v_850", "v_600", "v_500", "v_200",
    "w_1000", "w_925", "w_850", "w_600", "w_500", "w_200", "w_700", "w_650",
    "vimfc", "tp_sum", "ivt", "tcrw", "olr"
]
    
    try:
        K.clear_session()
        print("Tensor session cleared.")
        
        # %%
        #Langkah 01 Persiapan Data
        ##########################################
        # Baca data observasi
        ##########################################
        namafolder = 'C:\FILE\PROYEK PROF EDDY\CURAH HUJAN DATA PAK YANTO\FIX TERBARU REVISI 2025/'
        namafile = "sum_bulanan_rainfall.txt"
        datasinloc = os.path.join(namafolder, namafile)
        dfch = pd.read_csv(datasinloc, sep="\t")
        dfchnonnan = dfch[~dfch["monthly_rainfall"].isna()]

        df_bulan_terpilih = dfchnonnan[
            (dfchnonnan["year"] == PilihTahun) & (dfchnonnan["month"] == PilihBulan)
        ].reset_index(drop=True)

        df_bulan_terpilih = df_bulan_terpilih.sample(frac=persentotaldata, random_state=42).reset_index(drop=True)


        # %%
        # AREA DARATAN ONLY
        # Folder penyimpanan file eksogen
        '''
        namaFolderEksogen = 'C:\\FILE\\PROYEK PROF EDDY\\DATA\\DATA ERA 5\\DOWNLOAD BULANAN (1985-2024)\\HASIL SIAP UPLOAD'
        df_eksogen_list = []

        # Buat daftar koordinat grid daratan Jawa
        daratan_jawa = []
        for lat in np.arange(-9.0, -5.75, 0.25):
            for lon in np.arange(104.75, 115.0, 0.25):
                if (
                    (-7.5 <= lat <= -6.0 and 105.0 <= lon <= 107.0) or
                    (-7.75 <= lat <= -6.0 and 107.0 <= lon <= 111.0) or
                    (-8.25 <= lat <= -6.5 and 111.0 <= lon <= 114.5)
                ):
                    daratan_jawa.append((round(lat, 2), round(lon, 2)))

        latlon_jawa = pd.DataFrame(daratan_jawa, columns=["lat", "lon"])

        def in_box(lat, lon, lat_min, lat_max, lon_min, lon_max):
            return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max

        # Kurangi area yang tidak termasuk daratan
        boxes_kurang = [
            (-6.5, -6.0, 108.75, 110.25),
            (-7.5, -7.0, 105.0, 106.0),
            (-6.25, -6.0, 110.5, 111.0),
            (-6.25, -6.0, 105.0, 105.5),
            (-6.5, -6.5, 112.25, 114.5)
        ]

        for lat_min, lat_max, lon_min, lon_max in boxes_kurang:
            latlon_jawa = latlon_jawa[~latlon_jawa.apply(
                lambda row: in_box(row['lat'], row['lon'], lat_min, lat_max, lon_min, lon_max),
                axis=1
            )]

        # Tambah area tambahan jika perlu
        boxes_tambah = [
            (-8.25, -8.0, 109.75, 111.0),
            (-8.75, -8.25, 113.0, 114.5),
            (-8.5, -8.5, 112.75, 112.75)
        ]

        box_tambah = []
        for lat_min, lat_max, lon_min, lon_max in boxes_tambah:
            for lat in np.arange(lat_min, lat_max + 0.001, 0.25):
                for lon in np.arange(lon_min, lon_max + 0.001, 0.25):
                    box_tambah.append((round(lat, 2), round(lon, 2)))

        # Gabungkan daratan utama dan tambahan
        latlon_jawa_final = pd.concat([
            latlon_jawa,
            pd.DataFrame(box_tambah, columns=["lat", "lon"])
        ], ignore_index=True).drop_duplicates().reset_index(drop=True)

        # Loop untuk membaca hanya file dalam rentang tahun yang diinginkan
        for file in os.listdir(namaFolderEksogen):
            if file.startswith("processed_era5jawa") and file.endswith(".xlsx"):
                try:
                    parts = file.split("_")
                    tahun = int(parts[2])
                    bulan = int(parts[3].split(".")[0])
                except (IndexError, ValueError):
                    continue

                if tahun == PilihTahun and bulan == PilihBulan:
                    fullPathFileEksogen = os.path.join(namaFolderEksogen, file)
                    dfeksogen = pd.read_excel(fullPathFileEksogen)
                    
                    # Select only columns that exist in kolom_terpilih
                    dfeksogen = dfeksogen.reindex(columns=kolom_terpilih)
                    
                    # Drop tp_sum column if exists
                    if "tp_sum" in dfeksogen.columns:
                        dfeksogen = dfeksogen.drop(columns=["tp_sum"])
                    
                    # Filter data berdasarkan latlon_jawa_final
                    dfeksogen = dfeksogen.merge(latlon_jawa_final, on=["lat", "lon"], how="inner").reset_index(drop=True)
                    
                    df_eksogen_list.append(dfeksogen)

        if df_eksogen_list:
            dfeksogen = pd.concat(df_eksogen_list, ignore_index=True)
            covariates = dfeksogen.iloc[:, 2:].values
            jumlah_kolom_eksogen = dfeksogen.shape[1] - 2
            print(f'Jumlah kolom Eksogen: {jumlah_kolom_eksogen}')
            templatesample = f"{jumlah_kolom_eksogen}var_{angkapersendata}%_stasiun"
            print('Daratan Only')
        else:
            print("Tidak ada file eksogen dalam rentang tahun yang dipilih.")
        '''

        #ALL MAP
        # Folder penyimpanan file eksogen
        namaFolderEksogen = 'C:\\FILE\\PROYEK PROF EDDY\\DATA\\DATA ERA 5\\DOWNLOAD BULANAN (1985-2024)\\HASIL SIAP UPLOAD'

        df_eksogen_list = []

        for file in os.listdir(namaFolderEksogen):
            if file.startswith("processed_era5jawa") and file.endswith(".xlsx"):
                try:
                    parts = file.split("_")
                    tahun = int(parts[2])
                    bulan = int(parts[3].split(".")[0])
                except (IndexError, ValueError):
                    continue

                if tahun == PilihTahun and bulan == PilihBulan:
                    fullPathFileEksogen = os.path.join(namaFolderEksogen, file)
                    df_eksogen = pd.read_excel(fullPathFileEksogen)
                    df_eksogen = df_eksogen[kolom_terpilih]
                    df_eksogen = df_eksogen.drop(columns=["tp_sum"])
                    df_eksogen_list.append(df_eksogen)

        if df_eksogen_list:
            dfeksogen = pd.concat(df_eksogen_list, ignore_index=True)

            covariates = dfeksogen.iloc[:, 2:].values
        else:
            print("Tidak ada file eksogen dalam rentang tahun yang dipilih.")

        jumlah_kolom_eksogen = len(dfeksogen.columns)-2
        print(f'Jumlah kolom Eksogen: {jumlah_kolom_eksogen}')
        print('Full No Masking')
        templatesample = f"{jumlah_kolom_eksogen}var_{angkapersendata}%_stasiun"
        # %%
        """**PLOT STASIUN**"""

        # Variabel hujan
        variabel_hujan = "monthly_rainfall"

        # Pastikan variabel ada
        if variabel_hujan in df_bulan_terpilih.columns:
            # Buat GeoDataFrame dari data titik
            gdf = gpd.GeoDataFrame(
                df_bulan_terpilih,
                geometry=gpd.points_from_xy(df_bulan_terpilih["long"], df_bulan_terpilih["lat"]),
                crs="EPSG:4326"  # WGS84
            )

            # Ubah ke CRS web mercator agar bisa pakai contextily
            gdf_web = gdf.to_crs(epsg=3857)

            # Buat plot
            fig, ax = plt.subplots(figsize=(10, 10))

            # Plot data tanpa legend
            gdf_plot = gdf_web.plot(
                column=variabel_hujan,
                ax=ax,
                cmap="rainbow",
                edgecolor="black",
                markersize=20,
                legend=False
            )

            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

            ax.set_title(f"Sebaran Stasiun {variabel_hujan.upper()} {PilihBulan}-{PilihTahun} {templatesample} Pulau Jawa", fontsize=12)
            ax.axis("off")


            sm = plt.cm.ScalarMappable(
                cmap="rainbow",
                norm=plt.Normalize(
                    vmin=gdf_web[variabel_hujan].min(),
                    vmax=gdf_web[variabel_hujan].max()
                )
            )
            sm._A = [] 

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1) 
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label(variabel_hujan)

            output_path = os.path.join(output_dir_png, f"Sebaran_Stasiun_{variabel_hujan.upper()}_{PilihBulan}-{PilihTahun}_{templatesample}_Pulau_Jawa.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        else:
            print(f"Variabel {variabel_hujan} tidak ditemukan dalam data.")


        # %%
        """**PLOT EKSOGEN**"""

        variabel_eksogen = "w_500"

        grid_size = 0.25

        if variabel_eksogen in dfeksogen.columns:
            def make_pixel(row):
                lon = row["lon"]
                lat = row["lat"]
                return box(
                    lon - grid_size / 2,
                    lat - grid_size / 2,
                    lon + grid_size / 2,
                    lat + grid_size / 2
                )

            dfeksogen["geometry"] = dfeksogen.apply(make_pixel, axis=1)
            gdf_eksogen = gpd.GeoDataFrame(dfeksogen, geometry="geometry", crs="EPSG:4326")
            dfeksogen = dfeksogen.drop(columns=["geometry"])

            gdf_eksogen_web = gdf_eksogen.to_crs(epsg=3857)

            fig, ax = plt.subplots(figsize=(10, 10))
            gdf_eksogen_web.plot(
                column=variabel_eksogen,
                ax=ax,
                cmap="rainbow",
                alpha=0.7,
                edgecolor="none",
                legend=False
            )

            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

            # Judul dan axis
            ax.set_title(f"Sebaran Eksogen {variabel_eksogen.upper()} {PilihBulan}-{PilihTahun} {templatesample} Pulau Jawa", fontsize=12)
            ax.axis("off")

            sm = plt.cm.ScalarMappable(
                cmap="rainbow",
                norm=plt.Normalize(
                    vmin=gdf_eksogen_web[variabel_eksogen].min(),
                    vmax=gdf_eksogen_web[variabel_eksogen].max()
                )
            )
            sm._A = []

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label(variabel_eksogen)

            output_path = os.path.join(output_dir_png, f"Sebaran_Eksogen_{variabel_eksogen.upper()}_{PilihBulan}-{PilihTahun}_{templatesample}_Pulau_Jawa.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        else:
            print(f"Variabel {variabel_eksogen} tidak ditemukan dalam data eksogen.")


        ch_lonlat=df_bulan_terpilih[['long','lat']].values 
        near = dfeksogen[['lon','lat' ]].values 

        tree = spatial.KDTree(list(zip(near[:,0].ravel(), near[:,1].ravel())))
        tree.data
        idx = tree.query(ch_lonlat)[1]

        ###############VER 1
        dfchnonnanmap = df_bulan_terpilih.assign(neighbor = idx)
        df_ch_map = dfchnonnanmap.groupby(['neighbor'])[['monthly_rainfall']].mean()
        ###############################

        ###############VER 2
        df_ch_map_seleksi = dfchnonnanmap.groupby('neighbor')[['monthly_rainfall']].mean()
        lat_long = dfchnonnanmap.groupby('neighbor')[['lat', 'long']].first()
        df_ch_map_seleksi = df_ch_map_seleksi.join(lat_long)

        points_to_remove = [
            (-7.9875, 111.805),
            (-8.078055556, 112.3247222),
            (-7.647322222, 111.892075),
            (-7.819722222, 112.8175),
            (-7.557777778, 110.1908333),
            (-7.388055556, 110.3941667),
            (-7.586983333, 111.9046639),
            (-7.545944444, 112.4659444),
            (-7.302222222, 110.1180556),
            (-7.211, 112.1859833)]
        
        mask = ~df_ch_map_seleksi.apply(lambda row: (row['lat'], row['long']) in points_to_remove, axis=1)
        df_ch_map_seleksi = df_ch_map_seleksi[mask]
        ############################################

        chvalue=df_ch_map_seleksi["monthly_rainfall"].values #tergantung seleksi point atau tidak
        z = chvalue[:,None]
        normalized_z=(z-min(z))/(max(z)-min(z))
        
        idx_new = df_ch_map_seleksi.index.values #tergantung seleksi point atau tidak
        matched_covariates = covariates[idx_new]

        # %%
        df_padan = pd.DataFrame(matched_covariates, columns=[f'covariate_{i+1}' for i in range(matched_covariates.shape[1])])
        df_padan['rainfall'] = normalized_z.ravel()

        coords = dfeksogen.iloc[idx_new][['lon', 'lat']].reset_index(drop=True)
        df_padan = pd.concat([coords, df_padan], axis=1)

        # %%
        #PLOT PADANAN
        # Data
        lons = df_padan['lon'].values
        lats = df_padan['lat'].values
        rain = df_padan['rainfall'].values
        idx_new = df_padan['idx_new'].values.astype(int)

        grid_size = 0.25

        # Membuat geometri kotak untuk setiap titik
        def make_pixel(lon, lat):
            return box(lon - grid_size / 2, lat - grid_size / 2, lon + grid_size / 2, lat + grid_size / 2)

        # Membuat GeoDataFrame
        geometries = [make_pixel(lon, lat) for lon, lat in zip(lons, lats)]
        gdf = gpd.GeoDataFrame({'geometry': geometries, 'rainfall': rain, 'idx_new': idx_new}, crs="EPSG:4326")

        # Ubah ke Web Mercator
        gdf_web = gdf.to_crs(epsg=3857)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plotkan geometri kotak berdasarkan nilai rainfall
        gdf_web.plot(column='rainfall', ax=ax, cmap='Blues', edgecolor='k', alpha=0.7, legend=False)

        # Menambahkan basemap
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=gdf_web['rainfall'].min(), vmax=gdf_web['rainfall'].max()))
        sm._A = []  # Membuat ScalarMappable tanpa data apa pun untuk menampilkan colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('Normalized Rainfall')

        # Tambahkan label idx_new di tengah kotak (dalam koordinat EPSG:3857)
        for geom, label in zip(gdf_web.geometry, gdf_web['idx_new']):
            x, y = geom.centroid.x, geom.centroid.y
            ax.text(x, y, label, ha='center', va='center', fontsize=5, fontweight='bold', color='black')

        # Fungsi bantu konversi ticks EPSG:3857 ke lon-lat
        def mercator_x_to_lon(x_vals):
            points = gpd.GeoSeries(gpd.points_from_xy(x_vals, np.zeros_like(x_vals)), crs="EPSG:3857")
            points = points.to_crs("EPSG:4326")
            return [pt.x for pt in points]

        def mercator_y_to_lat(y_vals):
            points = gpd.GeoSeries(gpd.points_from_xy(np.zeros_like(y_vals), y_vals), crs="EPSG:3857")
            points = points.to_crs("EPSG:4326")
            return [pt.y for pt in points]

        # Ambil ticks axis saat ini
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()

        # Konversi ticks ke lat/lon
        xtick_labels = [f"{x:.2f}°" for x in mercator_x_to_lon(xticks)]
        ytick_labels = [f"{y:.2f}°" for y in mercator_y_to_lat(yticks)]

        # Set ticks dan labels baru
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)

        # Tambahkan grid garis
        ax.grid(which='both', linestyle='--', color='gray', alpha=0.5)

        # Judul
        ax.set_title('Distribusi Rainfall', fontsize=12)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Jangan matikan axis supaya label muncul
        ax.axis('on')

        # Tampilkan plot
        df_padan.to_csv(os.path.join(output_dir_png, f"DF_PADAN_{PilihBulan}_{PilihTahun}_{templatesample}.csv"), index=False)
        plt.savefig(os.path.join(output_dir_png, f"Plot_DF_PADAN_AfterSelection_{PilihBulan}-{PilihTahun}_{templatesample}_Pulau_Jawa.png"), dpi=300, bbox_inches='tight')

        # %%
        normalized_covariates =np.copy(covariates) 
        for i in range(covariates.shape[1]):
            normalized_covariates[:,i]=(((covariates[:,i]- min(matched_covariates[:,i])) / (max(matched_covariates[:,i])- min(matched_covariates[:,i]))))
        #########################################
        # %%
        pd.DataFrame(normalized_covariates)

        # %%
        (pd.DataFrame(normalized_covariates [idx_new,:])).describe() #DARATAN

        # %%
        lon = dfeksogen.values[:,1] #basic
        lat = dfeksogen.values[:,0] #basic

        normalized_lon = (lon-min(lon[idx_new]))/(max(lon[idx_new])-min(lon[idx_new]))#tsaqib
        normalized_lat = (lat-min(lat[idx_new]))/(max(lat[idx_new])-min(lat[idx_new]))#tsaqib
        N = lon.shape[0]

        num_basis = [10**2,19**2,37**2]
        knots_1dx = [np.linspace(0, 1, int(i**0.5)) for i in num_basis]
        knots_1dy = [np.linspace(0, 1, int(i**0.5)) for i in num_basis]

        basis_size = 0
        phi = np.zeros((N, sum(num_basis)))
        for res in range(len(num_basis)):
            theta = 1 / (np.sqrt(float(num_basis[res])) * 2.5)
            knots_x, knots_y = np.meshgrid(knots_1dx[res],knots_1dy[res])
            knots = np.column_stack((knots_x.flatten(),knots_y.flatten()))
            for i in range(num_basis[res]):
                dx = normalized_lon - knots[i, 0]
                dy = normalized_lat - knots[i, 1]
                d = ((dx ** 2 + dy ** 2) ** 0.5) / theta

                for j in range(len(d)):
                    if d[j] >= 0 and d[j] <= 1:
                        phi[j,i + basis_size] = (1-d[j])**6 * (35 * d[j]**2 + 18 * d[j] + 3)/3
                    else:
                        phi[j,i + basis_size] = 0
            basis_size = basis_size + num_basis[res]

        #################################################
        idx_zero = np.array([], dtype=int)
        for i in range(phi.shape[1]):
            if sum(phi[:,i]!=0)==0:
                idx_zero = np.append(idx_zero,int(i))

        phi_reduce = np.delete(phi,idx_zero,1)
        ############################################################

        phi_obs = phi_reduce [idx_new,:] #basic

        s_obs = np.vstack((normalized_lon[idx_new],normalized_lat[idx_new])).T #ver basic

        X = normalized_covariates [idx_new,:] #ver basic
        normalized_X = X
        N_obs = phi_obs.shape[0]
        #################################################################

        def deep_model(model, X_train, y_train, X_valid, y_valid, data_type,threshold_mae):

        #    Function to train a multi-class model. The number of epochs and
        #    batch_size are set by the constants at the top of the
        #    notebook.

        #    Parameters:
        #        model : model with the chosen architecture
        #        X_train : training features
        #        y_train : training target
        #        X_valid : validation features
        #        Y_valid : validation target
        #    Output:
        #        model training history


            # callbacks = [EarlyStopping(monitor='val_loss', patience= 30),
            #      ModelCheckpoint(filepath='model_real-50k_all_time.h5', monitor='val_loss', save_best_only=True)]

            if data_type == 'continuous':
                model.compile(optimizer='RMSprop', loss='mse', metrics=['mse','mae'])
            if data_type == 'discrete':
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), verbose=1,epochs=1)
            old=history.history['mae'][0]
            print(history.history)
            print(old)
            count=0
            nepoch=1
            while (history.history['mae'][0]>threshold_mae) and count<50:
                # history = model.fit(X_train
                #                , y_train
                #                #, epochs=NB_START_EPOCHS
                #                #
                #                , validation_data=(X_valid, y_valid)
                #                , verbose=0)

                history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), verbose=1,epochs=1)
                beda=np.abs(old-history.history['mae'][0])
                if beda<0.00001:
                    count=count+1
                old=history.history['mae'][0]
                print(beda,",",old)
                nepoch=nepoch+1
            #print(beda,",",old)
            return history,nepoch
        #########################################################################

        #Langkah ke 2 Model Arsitektur

        p = covariates.shape[1] + phi_reduce.shape[1] #ver basic
        #p = X.shape[1] + phi_reduce.shape[1] #edittsaqib

        #p = covariates.shape[1]
        model = Sequential()
        model.add(Dense(64, input_dim = p,  kernel_initializer='he_uniform', activation='relu'))
        #model.add(Dropout(rate=0.5))
        #model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        #model.add(Dropout(rate=0.5))
        #model.add(Dense(100, activation='relu'))
        #model.add(BatchNormalization())
        model.add(Dense(1, activation='relu'))

        #Langkah ke 3 Learning Model
        from datetime import datetime
        inputs = np.hstack((normalized_X,phi_obs)) #AMBIL PADANAN SAJA

        #inputs = normalized_X
        targets = z
        start_time = datetime.now()
        # do your work here

        #maenya=0.2
        history1,npc=deep_model(model, inputs, targets[:,0:1], inputs, targets[:,0:1],'continuous',maenya)
        # do your work here
        end_time = datetime.now()
        waktunya=str(maenya)+',{}'.format(end_time - start_time)+","+str(npc)+","+str(history1.history["mae"][0])
        print(str(maenya)+',Duration: {}'.format(end_time - start_time)+","+str(npc)+","+str(history1.history["mae"][0]))


        # %%
        model.summary()

        # %%
        normalized_X_pred = normalized_covariates #verbasic

        X_RBF_pred = np.hstack((normalized_X_pred,phi_reduce))
        CH_pred = model.predict(X_RBF_pred)

        # %%
        s = np.vstack((lon,lat)).T

        CH_pred_combine = np.concatenate((s,CH_pred),axis=1) #ver basic

        CH_pred_df=pd.DataFrame(CH_pred_combine)
        CH_pred_df.columns = ['longitude', 'latitude', 'ch_pred']

        namaFolderSimpan=output_dir_png
        namaFileSimpan=f"HASIL_ch_pred_{PilihBulan}-{PilihTahun}_{templatesample}.csv"
        fullPathFileSimpan=os.path.join(namaFolderSimpan,namaFileSimpan)
        CH_pred_df.to_csv(fullPathFileSimpan)

        boxes = []
        for _, row in CH_pred_df.iterrows():
            boxes.append(box(
                row['longitude'] - 0.125,  
                row['latitude'] - 0.125,  
                row['longitude'] + 0.125,  
                row['latitude'] + 0.125    
            ))

        gdf = gpd.GeoDataFrame({
            'geometry': boxes,
            'rainfall': CH_pred_df['ch_pred']
        }, crs="EPSG:4326")

        gdf_web = gdf.to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(10, 8))

        gdf_web.plot(
            column='rainfall',
            ax=ax,
            cmap='rainbow',
            vmin=0,
            vmax=2000,
            edgecolor='black',
            linewidth=0.1,
            alpha=0.7,
            legend=False
        )

        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        sm = plt.cm.ScalarMappable(
            cmap='rainbow',
            norm=plt.Normalize(vmin=0, vmax=2000)
        )
        sm._A = []
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Curah Hujan (mm)")

        ax.set_title(f"CH-Pred {PilihBulan}-{PilihTahun}_{templatesample}", fontsize=12)
        ax.axis('off')

        os.makedirs(output_dir_png, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir_png, f"plot_DK Pred_{PilihBulan}-{PilihTahun}_{templatesample}.png"),
            dpi=300,
            bbox_inches='tight'
        )
        
        print(f"Successfully processed month {PilihBulan}")
        
    except Exception as e:
        print(f"Error processing month {PilihBulan}: {str(e)}")
        continue

    import time
    time.sleep(30)

print("Processing complete for all months!")
