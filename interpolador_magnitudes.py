import re 

def extraer_orden(nombre_archivo):
        match = re.search(r'm(\d+)', nombre_archivo)
        return int(match.group(1)) if match else 0

def interpolar_espectro(ruta_carpeta, patron="*AIR_modificado.par_tac.fits", num_puntos=50000, plot=True):
    from scipy.interpolate import interp1d
    from astropy.io import fits
    import os
    import glob
    import matplotlib.pyplot as plt
    import numpy as np
    

    
    if not os.path.exists(ruta_carpeta):
        raise FileNotFoundError(f"La carpeta {ruta_carpeta} no existe")

    patron_path = os.path.join(ruta_carpeta, patron)
    archivos = glob.glob(patron_path)
    print(f"procesando archivos: {archivos}")

    # ordenar por ordenes de echelle
    archivos.sort(key=extraer_orden)
    print(f"archivos ordenados por orden de echelle: {archivos}")
    
    if not archivos:
        print(f"No se encontraron archivos en {ruta_carpeta}")
        return None, None

    print(f"Leyendo {len(archivos)} archivos FITS...")

    lambdas = []
    fluxes = []

    for archivo in archivos:
        with fits.open(archivo) as hdul:
            datos = hdul[1].data
            longitud_onda = datos['LAMBDA']  # a Angstroms
            flujo = datos['FLUX']
            lambdas.append(longitud_onda)
            fluxes.append(flujo)

    # Grilla común
    w_min = min(np.min(l) for l in lambdas)
    w_max = max(np.max(l) for l in lambdas)
    long_onda_nueva = np.linspace(w_min, w_max, num_puntos)

    flux_acumulado = np.zeros_like(long_onda_nueva)
    contador = np.zeros_like(long_onda_nueva)  # cuántos órdenes REALES cubren cada punto

    for lam, flx in zip(lambdas, fluxes):
        lam_min, lam_max = np.min(lam), np.max(lam)

        # Máscara: puntos de la grilla que caen DENTRO del rango real de este orden
        mascara_real = (long_onda_nueva >= lam_min) & (long_onda_nueva <= lam_max)

        # Interpolar solo donde hay datos reales (sin fill_value engañoso)
        f_interp = interp1d(lam, flx, kind='linear', bounds_error=False, fill_value=np.nan)
        flux_interp = f_interp(long_onda_nueva)

        # Acumular solo en zonas con cobertura real
        validos = mascara_real & ~np.isnan(flux_interp)
        flux_acumulado[validos] += flux_interp[validos]
        contador[validos] += 1

    # Promedio donde hay datos reales, 1 en los gaps
    flux_promedio = np.where(contador > 0, flux_acumulado / contador, 1.0)

    print("Se leyeron e interpolaron correctamente los espectros")

    if plot:
        fig, ax = plt.subplots(figsize=(12, 5))
        
        ax.plot(long_onda_nueva, flux_promedio, lw=0.5, label='Espectro combinado', color = 'black')
        ax.set_xlabel('Longitud de onda (Å)')
        ax.set_ylabel('Flujo normalizado')
        ax.set_title('Espectro interpolado (órdenes echelle)')
        ax.legend()
        # plotear los espectros originales encima 
        for lam, flx in zip(lambdas, fluxes):
            ax.plot(lam, flx, lw=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Guardar FITS
    hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='LAMBDA', format='E', array=long_onda_nueva / 1e4),
        fits.Column(name='CFLUX',  format='E', array=flux_promedio)
    ])
    nuevo_archivo = os.path.join(ruta_carpeta, f'{ruta_carpeta[-6:-1]}interpolado.par_tac.fits')
    hdu.writeto(nuevo_archivo, overwrite=True)
    print(f"Archivo FITS guardado en {nuevo_archivo}")

    return long_onda_nueva, flux_promedio
