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

######################## ISI CUSTOM ###############################3
PilihTahun = 1987  # PILIH TAHUN
for PilihBulan in range(1 , 13): #PILIH BULAN RANGE
    print(f"\nProcessing Month {PilihBulan} Year {PilihTahun}")
    
    #maenya=0.8 
    persentotaldata= 1 #RANGE % STASIUN (0.0 - 1.0)
    angkapersendata = persentotaldata*100
    templatesample = f"51var_{angkapersendata}%_stasiun"

    namafolder = '\CH\\' #CH_HUJAN
    namafile = "sum_bulanan_rainfall.txt"
    
    namaFolderEksogen = '\ERA5\\' #FOLDER_EKSOGEN

    folder_save_hasil = "\FOLDER SIMPAN\\" #FOLDER FILE DISIMPAN SAVE PNG, SAVE HASIL
    output_dir_png = folder_save_hasil

    kolom_terpilih = [ #pilih variabel kolom dari file /ERA5/
    "lat", "lon", "cape", "hcc", "mcc", "msl", "slhf", "sshf", "t2m", "tclw", "tcwv",
    "q_1000", "q_925", "q_850", "q_600", "q_500", "q_200",
    "r_1000", "r_925", "r_850", "r_600", "r_500", "r_200",
    "t_1000", "t_925", "t_850", "t_600", "t_500", "t_200",
    "u_1000", "u_925", "u_850", "u_600", "u_500", "u_200",
    "v_1000", "v_925", "v_850", "v_600", "v_500", "v_200",
    "w_1000", "w_925", "w_850", "w_600", "w_500", "w_200", "w_700", "w_650",
    "vimfc", "tp_sum", "ivt", "tcrw", "olr"
]
###################################################################################
    
    # OTOMATIS
    try:
        K.clear_session()
        print("Tensor session cleared.")
        
        # %%
        #Langkah 01 Persiapan Data
        ##########################################
        # Baca data observasi
        ##########################################
        datasinloc = os.path.join(namafolder, namafile)
        dfch = pd.read_csv(datasinloc, sep="\t")
        dfchnonnan = dfch[~dfch["monthly_rainfall"].isna()]

        df_bulan_terpilih = dfchnonnan[
            (dfchnonnan["year"] == PilihTahun) & (dfchnonnan["month"] == PilihBulan)
        ].reset_index(drop=True)

        df_bulan_terpilih = df_bulan_terpilih.sample(frac=persentotaldata, random_state=42).reset_index(drop=True)
        # %%
        # AREA DARATAN ONLY
        '''
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

        if variabel_hujan in df_bulan_terpilih.columns:
            gdf = gpd.GeoDataFrame(
                df_bulan_terpilih,
                geometry=gpd.points_from_xy(df_bulan_terpilih["long"], df_bulan_terpilih["lat"]),
                crs="EPSG:4326"
            )

            gdf_web = gdf.to_crs(epsg=3857)
            fig, ax = plt.subplots(figsize=(10, 10))

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
        dfchnonnanmap = df_bulan_terpilih.assign(neighbor = idx)
        df_ch_map = dfchnonnanmap.groupby(['neighbor'])[['monthly_rainfall']].mean()
        df_ch_map_class0 = dfchnonnanmap.groupby('neighbor')['monthly_rainfall'].mean()
        chvalue=df_ch_map["monthly_rainfall"].values
        z = chvalue[:,None]
        normalized_z=(z-min(z))/(max(z)-min(z))
        idx_new = df_ch_map.index.values

        matched_covariates = covariates[idx_new]

        ch_lonlat = df_bulan_terpilih[['long', 'lat']].values

        # Lokasi lat lon untuk data eksogen

        #TSAQIBB
        df_ch_map_class = pd.cut(df_ch_map_class0,bins=[0,12.1,35.5],labels=["0","1"])
        df_ch_map_class= df_ch_map_class.astype('float').fillna(0)
        ch_map_class = np.array( df_ch_map_class.values, dtype=int)
        z_class = ch_map_class[:,None]

        idx_new = df_ch_map.index.values

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
        grid_size = 0.25

        def make_pixel(lon, lat):
            return box(lon - grid_size / 2, lat - grid_size / 2, lon + grid_size / 2, lat + grid_size / 2)

        geometries = [make_pixel(lon, lat) for lon, lat in zip(lons, lats)]
        gdf = gpd.GeoDataFrame({'geometry': geometries, 'rainfall': rain}, crs="EPSG:4326")

        gdf_web = gdf.to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(10, 8))

        gdf_web.plot(column='rainfall', ax=ax, cmap='Blues', edgecolor='k', alpha=0.7, legend=False)

        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=gdf_web['rainfall'].min(), vmax=gdf_web['rainfall'].max()))
        sm._A = [] 
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('Normalized Rainfall')

        ax.set_title('Distribusi Rainfall', fontsize=12)
        ax.axis('off')

        #save plot
        plt.savefig(os.path.join(output_dir_png, f"Plot_PADANAN_STasiun_ERA5_{PilihBulan}-{PilihTahun}_{templatesample}.png"),pi=300,bbox_inches='tight')


        # %%
        #df_padan.to_csv(os.path.join(folder_save_hasil, f"DF_PADAN_{PilihBulan}_{PilihTahun}_{templatesample}.csv"), index=False)

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
        inputs = np.concatenate((phi_obs, normalized_X), axis=1)
        inputs_base = np.concatenate((s_obs, normalized_X), axis=1)

        def deep_model(model, X_train, y_train, X_valid, y_valid, data_type):
            '''
            Function to train a multi-class model. The number of epochs and
            batch_size are set by the constants at the top of the
            notebook.

            Parameters:
                model : model with the chosen architecture
                X_train : training features
                y_train : training target
                X_valid : validation features
                Y_valid : validation target
            Output:
                model training history
            '''
            if data_type == 'continuous':
                model.compile(optimizer='adam'
                            , loss='mse'
                            , metrics=['mse','mae','mape'])
                
            if data_type == 'discrete':
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            history = model.fit(X_train
                            , y_train
                            , epochs=NB_START_EPOCHS
                            , batch_size=BATCH_SIZE
                            , validation_data=(X_valid, y_valid)
                            , verbose=0)
            return history

        def test_model(model, X_train, y_train, X_test, y_test, epoch_stop):
            '''
            Function to test the model on new data after training it
            on the full training data with the optimal number of epochs.

            Parameters:
                model : trained model
                X_train : training features
                y_train : training target
                X_test : test features
                y_test : test target
                epochs : optimal number of epochs
            Output:
                test accuracy and test loss
            '''
            model.fit(X_train
                    , y_train
                    , epochs=epoch_stop
                    , batch_size=BATCH_SIZE
                    , verbose=0)
            results = model.evaluate(X_test, y_test, verbose=0)
            return results

        def optimal_epoch(model_hist):
            '''
            Function to return the epoch number where the validation loss is
            at its minimum

            Parameters:
                model_hist : training history of model
            Output:
                epoch number with minimum validation loss
            '''
            min_epoch = np.argmin(model_hist.history['val_loss']) + 1
            return min_epoch
        #########################################################################

        #Langkah ke 2 Model Arsitektur

        p = covariates.shape[1] + phi_reduce.shape[1]
        model = Sequential()
        model.add(Dense(100, input_dim = p,  kernel_initializer='he_uniform', activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(BatchNormalization())
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(100, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='linear'))


        # In[46]:


        # DeepKriging model for categorical data
        model_class = Sequential()
        model_class.add(Dense(100, input_dim = p,  kernel_initializer='he_uniform', activation='relu'))
        model_class.add(Dropout(rate=0.5))
        model_class.add(BatchNormalization())
        model_class.add(Dense(100, activation='relu'))
        model_class.add(Dropout(rate=0.5))
        model_class.add(Dense(100, activation='relu'))
        model_class.add(BatchNormalization())
        model_class.add(Dense(1, activation='sigmoid'))


        # In[48]:

        # Baseline DNN only with covariates and coordinates
        p_base = covariates.shape[1] + s_obs.shape[1]
        # Neural network
        model_base = Sequential()
        model_base.add(Dense(100, input_dim=p_base,  kernel_initializer='he_uniform', activation='relu'))
        model_base.add(Dropout(rate=0.5))
        model_base.add(BatchNormalization())
        model_base.add(Dense(100, activation='relu'))
        model_base.add(Dropout(rate=0.5))
        model_base.add(Dense(100, activation='relu'))
        model_base.add(BatchNormalization())
        model_base.add(Dense(1, activation='linear'))

        # In[50]:

        # Baseline DNN for classification
        model_base_class = Sequential()
        model_base_class.add(Dense(100, input_dim=p_base,  kernel_initializer='he_uniform', activation='relu'))
        model_base_class.add(Dropout(rate=0.5))
        model_base_class.add(BatchNormalization())
        model_base_class.add(Dense(100, activation='relu'))
        model_base_class.add(Dropout(rate=0.5))
        model_base_class.add(Dense(100, activation='relu'))
        model_base_class.add(BatchNormalization())
        model_base_class.add(Dense(1, activation='sigmoid'))

        p_cov = covariates.shape[1]
        model_cov = Sequential()
        model_cov.add(Dense(100, input_dim = p_cov,  kernel_initializer='he_uniform', activation='relu'))
        model_cov.add(Dropout(rate=0.5))
        model_cov.add(BatchNormalization())
        model_cov.add(Dense(100, activation='relu'))
        model_cov.add(Dropout(rate=0.5))
        model_cov.add(Dense(100, activation='relu'))
        model_cov.add(BatchNormalization())
        model_cov.add(Dense(1, activation='linear'))

        model_cov_class = Sequential()
        model_cov_class.add(Dense(100, input_dim = p_cov,  kernel_initializer='he_uniform', activation='relu'))
        model_cov_class.add(Dropout(rate=0.5))
        model_cov_class.add(BatchNormalization())
        model_cov_class.add(Dense(100, activation='relu'))
        model_cov_class.add(Dropout(rate=0.5))
        model_cov_class.add(Dense(100, activation='relu'))
        model_cov_class.add(BatchNormalization())
        model_cov_class.add(Dense(1, activation='sigmoid'))
        # ### Run cross-validation

        # In[51]:
        from sklearn.model_selection import KFold
        NB_START_EPOCHS = 200  # Number of epochs we usually start to train with
        BATCH_SIZE = 64  # Size of the batches used in the mini-batch gradient descent

        num_folds = 10
        kfold = KFold(n_splits=num_folds, shuffle=False)
        #kfold = KFold(n_splits=num_folds, shuffle=True, random_state = 123)

        fold_no = 1
        inputs = np.hstack((normalized_X,phi_obs))
        inputs_base = np.hstack((normalized_X,s_obs))
        inputs_cov = normalized_X
        targets = z
        targets_class = z_class
        mse_per_fold = []
        mse_per_fold_base = []
        mse_per_fold_cov = []
        mae_per_fold = []
        mae_per_fold_base = []
        mae_per_fold_cov = []
        mape_per_fold = []
        mape_per_fold_base = []
        mape_per_fold_cov = []
        acc_per_fold = []
        acc_per_fold_base = []
        acc_per_fold_cov = []

        for train_idx, test_idx in kfold.split(inputs, targets):
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ...')
            history = deep_model(model, inputs[train_idx], targets[train_idx,0:1], inputs[test_idx], targets[test_idx,0:1],'continuous')
            history_base = deep_model(model_base, inputs_base[train_idx], targets[train_idx,0:1], inputs_base[test_idx], targets[test_idx,0:1],'continuous')
            history_cov = deep_model(model_cov, inputs_cov[train_idx], targets[train_idx,0:1], inputs_cov[test_idx], targets[test_idx,0:1],'continuous')
            ## Classification
            history_class = deep_model(model_class, inputs[train_idx], targets_class[train_idx,0:1], inputs[test_idx], targets_class[test_idx,0:1],'discrete')
            history_base_class = deep_model(model_base_class, inputs_base[train_idx], targets_class[train_idx,0:1], inputs_base[test_idx], targets_class[test_idx,0:1],'discrete')
            history_cov_class = deep_model(model_cov_class, inputs_cov[train_idx], targets_class[train_idx,0:1], inputs_cov[test_idx], targets_class[test_idx,0:1],'discrete')
            
            model_optim = 200#optimal_epoch(history)
            model_optim_base = 200#optimal_epoch(history_base)
            model_optim_cov = 200#optimal_epoch(history_base)
            
            result = test_model(model, inputs[train_idx], targets[train_idx,0:1], inputs[test_idx], targets[test_idx,0:1], model_optim)
            result_base = test_model(model_base, inputs_base[train_idx], targets[train_idx,0:1], inputs_base[test_idx], targets[test_idx,0:1], model_optim_base)
            result_cov = test_model(model_cov, inputs_cov[train_idx], targets[train_idx,0:1], inputs_cov[test_idx], targets[test_idx,0:1], model_optim_cov)
            
            scores = result
            scores_base = result_base
            scores_cov = result_cov

            print(f'The performance of DeepKriging: MSE = {scores[1]}, MAE = {scores[2]},MAPE = {scores[3]}')
            print(f'The performance of classical DNN: MSE = {scores_base[1]}, MAE = {scores_base[2]},MAPE = {scores_base[3]}')
            print(f'The performance of covariate only: MSE = {scores_cov[1]}, MAE = {scores_cov[2]},MAPE = {scores_cov[3]}')

            model_optim_class = 200#optimal_epoch(history_class)
            model_optim_base_class = 200#optimal_epoch(history_base_class)
            model_optim_cov_class = 200#optimal_epoch(history_base_class)

            result_class = test_model(model_class, inputs[train_idx], targets_class[train_idx,0:1], inputs[test_idx], targets_class[test_idx,0:1], model_optim_class)
            result_base_class = test_model(model_base_class, inputs_base[train_idx], targets_class[train_idx,0:1], inputs_base[test_idx], targets_class[test_idx,0:1], model_optim_base_class)
            result_cov_class = test_model(model_cov_class, inputs_cov[train_idx], targets_class[train_idx,0:1], inputs_cov[test_idx], targets_class[test_idx,0:1], model_optim_cov_class)
            
            scores_class = result_class
            scores_base_class = result_base_class
            scores_cov_class = result_cov_class

            print(f'The performance of DeepKriging: accuracy = {scores_class[1]}')
            print(f'The performance of classical DNN: accuracy = {result_base_class[1]}')
            print(f'The performance of covariate only: accuracy = {result_cov_class[1]}')

            fold_no = fold_no + 1
            acc_per_fold.append(scores_class[1])
            acc_per_fold_base.append(scores_base_class[1])
            acc_per_fold_cov.append(scores_cov_class[1])

            mse_per_fold.append(scores[1])
            mse_per_fold_base.append(scores_base[1])
            mse_per_fold_cov.append(scores_cov[1])

            mae_per_fold.append(scores[2])
            mae_per_fold_base.append(scores_base[2])
            mae_per_fold_cov.append(scores_cov[2])

            mape_per_fold.append(scores[3])
            mape_per_fold_base.append(scores_base[3])
            mape_per_fold_cov.append(scores_cov[3])

        # In[70]:
        ##Summerize the results
        print("result summary")
        print(np.mean(mse_per_fold))
        print(np.std(mse_per_fold))
        print(np.mean(mse_per_fold_base))
        print(np.std(mse_per_fold_base))
        print(np.mean(mse_per_fold_cov))
        print(np.std(mse_per_fold_cov))

        print(np.mean(mae_per_fold))
        print(np.std(mae_per_fold))
        print(np.mean(mae_per_fold_base))
        print(np.std(mae_per_fold_base))
        print(np.mean(mae_per_fold_cov))
        print(np.std(mae_per_fold_cov))

        print(np.mean(mape_per_fold))
        print(np.std(mape_per_fold))
        print(np.mean(mape_per_fold_base))
        print(np.std(mape_per_fold_base))
        print(np.mean(mape_per_fold_cov))
        print(np.std(mape_per_fold_cov))

        print(np.mean(acc_per_fold))
        print(np.std(acc_per_fold))
        print(np.mean(acc_per_fold_base))
        print(np.std(acc_per_fold_base))
        print(np.mean(acc_per_fold_cov))
        print(np.std(acc_per_fold_cov))
        # Data yang akan disimpan dari hasil summary metrik
        summary_data = {
            'Metric': ['Mean MSE', 'Std MSE', 'Mean MSE Base', 'Std MSE Base','Mean MSE Cov', 'Std MSE Cov',
                    'Mean MAE', 'Std MAE', 'Mean MAE Base', 'Std MAE Base','Mean MAE Cov', 'Std MAE Cov',
                    'Mean MAPE', 'Std MAPE', 'Mean MAPE Base', 'Std MAPE Base','Mean MAPE Cov', 'Std MAPE Cov',
                    'Mean Accuracy', 'Std Accuracy', 'Mean Accuracy Base', 'Std Accuracy Base','Mean Accuracy Cov', 'Std Accuracy Cov'],
            'Value': [
                np.mean(mse_per_fold),
                np.std(mse_per_fold),
                np.mean(mse_per_fold_base),
                np.std(mse_per_fold_base),
                np.mean(mse_per_fold_cov),
                np.std(mse_per_fold_cov),

                np.mean(mae_per_fold),
                np.std(mae_per_fold),
                np.mean(mae_per_fold_base),
                np.std(mae_per_fold_base),
                np.mean(mae_per_fold_cov),
                np.std(mae_per_fold_cov),

                np.mean(mape_per_fold),
                np.std(mape_per_fold),
                np.mean(mape_per_fold_base),
                np.std(mape_per_fold_base),
                np.mean(mape_per_fold_cov),
                np.std(mape_per_fold_cov),

                np.mean(acc_per_fold),
                np.std(acc_per_fold),
                np.mean(acc_per_fold_base),
                np.std(acc_per_fold_base),
                np.mean(acc_per_fold_cov),
                np.std(acc_per_fold_cov)
            ]
        }

        # Membuat DataFrame untuk summary metrik
        df_summary = pd.DataFrame(summary_data)

        # Data hasil k-fold yang akan disimpan
        kfold_data = {
            'Fold': list(range(1, fold_no)),  # Menyusun nomor fold
            'DeepKriging Accuracy': acc_per_fold,
            'Classical DNN Accuracy': acc_per_fold_base,
            'Covatiates Only Accuracy':acc_per_fold_cov,
            'DeepKriging MSE': mse_per_fold,
            'Classical DNN MSE': mse_per_fold_base,
            'Covatiates Only MSE': mse_per_fold_cov,
            'DeepKriging MAE': mae_per_fold,
            'Classical DNN MAE': mae_per_fold_base,
            'Covatiates Only MAE': mae_per_fold_cov,
            'DeepKriging MAPE': mape_per_fold,
            'Classical DNN MAPE': mape_per_fold_base,
            'Covatiates Only MAPE': mape_per_fold_cov,
        }

        # Membuat DataFrame untuk k-fold results
        df_kfold = pd.DataFrame(kfold_data)

        # Menyimpan kedua DataFrame ke dalam file Excel yang sama, dengan spasi kolom
        excel_filename = f'C:\\FILE\\PROYEK PROF EDDY\\DATA\\DK_MERGE_TSAQIB\\KFold\\summary_metrics_newver_{PilihTahun}_{PilihBulan}.xlsx'

        with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
            # Menyimpan Summary Metrics dengan judul tabel
            df_summary.to_excel(writer, sheet_name='Combined Results', index=False, startrow=2, startcol=0)
            worksheet = writer.sheets['Combined Results']
            worksheet.write('A1', f'Summary of Metrics {PilihTahun}-{PilihBulan}')  # Judul Tabel
            
            # Menambahkan beberapa kolom kosong sebelum menyimpan KFold Results
            startcol_for_kfold = len(df_summary.columns) + 1  # Memberikan jarak 3 kolom kosong
            df_kfold.to_excel(writer, sheet_name='Combined Results', index=False, startrow=2, startcol=startcol_for_kfold)
            worksheet = writer.sheets['Combined Results']
            worksheet.write(f'A{2}', f'K-Fold Cross-Validation Results {PilihTahun}-{PilihBulan}')  # Judul Tabel KFold

        print(f'File Excel "{excel_filename}" berhasil disimpan.')

    except Exception as e:
        print(f"Error processing month {PilihBulan}: {str(e)}")
        continue
    import time
    time.sleep(30)

print("Processing complete for all months!")

'''
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

'''
