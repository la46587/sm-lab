import cv2
import numpy as np
import matplotlib.pyplot as plt


class Data:
    def __init__(self):
        self.Y = None
        self.Cb = None
        self.Cr = None


def chroma_subsampling(L, subsampling):
    if subsampling == '4:4:4':
        return L
    elif subsampling == '4:2:2':
        return L[:, ::2]
    elif subsampling == '4:4:0':
        return L[::2, :]
    elif subsampling == '4:2:0':
        return L[::2, ::2]
    elif subsampling == '4:1:1':
        return L[:, ::4]
    elif subsampling == '4:1:0':
        return L[::2, ::4]
    return L


def chroma_resampling(L, subsampling):
    if subsampling == '4:4:4':
        return L
    elif subsampling == '4:2:2':
        L = np.repeat(L, 2, axis=1)
    elif subsampling == '4:4:0':
        L = np.repeat(L, 2, axis=0)
    elif subsampling == '4:2:0':
        L = np.repeat(L, 2, axis=1)
        L = np.repeat(L, 2, axis=0)
    elif subsampling == '4:1:1':
        L = np.repeat(L, 4, axis=1)
    elif subsampling == '4:1:0':
        L = np.repeat(L, 4, axis=1)
        L = np.repeat(L, 2, axis=0)
    return L


def frame_image_to_class(frame, subsampling):
    Frame_class = Data()
    Frame_class.Y = frame[:, :, 0].astype(int)
    Frame_class.Cb = chroma_subsampling(frame[:, :, 2].astype(int), subsampling)
    Frame_class.Cr = chroma_subsampling(frame[:, :, 1].astype(int), subsampling)
    return Frame_class


def frame_layers_to_image(Y, Cr, Cb, subsampling):
    Cb = chroma_resampling(Cb, subsampling)
    Cr = chroma_resampling(Cr, subsampling)
    return np.dstack([Y, Cr, Cb]).clip(0, 255).astype(np.uint8)


def compress_KeyFrame(Frame_class):
    KeyFrame = Data()
    KeyFrame.Y = Frame_class.Y
    KeyFrame.Cb = Frame_class.Cb
    KeyFrame.Cr = Frame_class.Cr
    return KeyFrame


def decompress_KeyFrame(KeyFrame):
    frame_image = frame_layers_to_image(KeyFrame.Y, KeyFrame.Cr, KeyFrame.Cb, subsampling)
    return frame_image


def compress_not_KeyFrame(Frame_class, KeyFrame, factor=8):
    Compress_data = Data()
    Compress_data.Y = (Frame_class.Y - KeyFrame.Y) / factor
    Compress_data.Cb = (Frame_class.Cb - KeyFrame.Cb) / factor
    Compress_data.Cr = (Frame_class.Cr - KeyFrame.Cr) / factor
    return Compress_data


def decompress_not_KeyFrame(Compress_data, KeyFrame, factor=8):
    Y = (Compress_data.Y * factor) + KeyFrame.Y
    Cb = (Compress_data.Cb * factor) + KeyFrame.Cb
    Cr = (Compress_data.Cr * factor) + KeyFrame.Cr
    frame_image = frame_layers_to_image(Y, Cr, Cb, subsampling)
    return frame_image


def plotDiffrence(ReferenceFrame, DecompressedFrame, ROI):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Porównanie obrazów w RGB')

    ref_region = ReferenceFrame[ROI[0]:ROI[1], ROI[2]:ROI[3], :]
    dec_region = DecompressedFrame[ROI[0]:ROI[1], ROI[2]:ROI[3], :]

    axs[0].imshow(ref_region)
    axs[0].set_title('Obraz Referencyjny')
    axs[0].axis('off')

    axs[1].imshow(dec_region)
    axs[1].set_title('Obraz Zdekompresowany')
    axs[1].axis('off')

    diff = ref_region.astype(float) - dec_region.astype(float)
    diff_abs = np.abs(diff)  # Obliczenie wartości bezwzględnej różnicy
    axs[2].imshow(diff_abs / diff_abs.max())  # Normalizacja dla lepszego wyświetlenia
    axs[2].set_title('Różnica')
    axs[2].axis('off')

    print(f"Minimalna różnica: {np.min(diff)}, Maksymalna różnica: {np.max(diff)}")

    plt.tight_layout()
    plt.show()


subsamples = ["4:4:4", "4:2:2", "4:4:0", "4:2:0", "4:1:1", "4:1:0"]

for subsample in subsamples:

    plik = "video.mp4"  # nazwa pliku
    ile = 500  # ile klatek odtworzyć? <0 - całość
    key_frame_counter = 50  # co która klatka ma być kluczowa i nie podlegać kompresji
    plot_frames = np.array([50, 100, 150, 200, 250, 300, 350, 450])  # automatycznie wyrysuj wykresy
    auto_pause_frames = np.array([25])  # automatycznie za pauzuj dla klatki
    subsampling = "4:1:0"  # parametry dla chroma subsampling
    dzielnik = 1  # dzielnik przy zapisie różnicy
    wyswietlaj_kaltki = True  # czy program ma wyświetlać klatki
    ROI = [[400, 500, 400, 500]]  # wyświetlane fragmenty (można podać kilka)

    cap = cv2.VideoCapture(plik)

    if ile < 0:
        ile = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cv2.namedWindow('Normal Frame')
    cv2.namedWindow('Decompressed Frame')

    compression_information = np.zeros((3, ile))

    for i in range(ile):
        ret, frame = cap.read()
        if wyswietlaj_kaltki:
            cv2.imshow('Normal Frame', frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        Frame_class = frame_image_to_class(frame, subsampling)
        if (i % key_frame_counter) == 0:  # pobieranie klatek kluczowych
            KeyFrame = compress_KeyFrame(Frame_class)
            cY = KeyFrame.Y
            cCb = KeyFrame.Cb
            cCr = KeyFrame.Cr
            Decompresed_Frame = decompress_KeyFrame(KeyFrame)
        else:  # kompresja
            Compress_data = compress_not_KeyFrame(Frame_class, KeyFrame)
            cY = Compress_data.Y
            cCb = Compress_data.Cb
            cCr = Compress_data.Cr
            Decompresed_Frame = decompress_not_KeyFrame(Compress_data, KeyFrame)

        compression_information[0, i] = (frame[:, :, 0].size - cY.size) / frame[:, :, 0].size
        compression_information[1, i] = (frame[:, :, 0].size - cCb.size) / frame[:, :, 0].size
        compression_information[2, i] = (frame[:, :, 0].size - cCr.size) / frame[:, :, 0].size
        if wyswietlaj_kaltki:
            cv2.imshow('Decompressed Frame', cv2.cvtColor(Decompresed_Frame, cv2.COLOR_YCrCb2BGR))

        if np.any(plot_frames == i):  # rysuj wykresy
            for r in ROI:
                plotDiffrence(frame, Decompresed_Frame, r)

        if np.any(auto_pause_frames == i):
            cv2.waitKey(-1)  # wait until any key is pressed

        k = cv2.waitKey(1) & 0xff

        if k == ord('q'):
            break
        elif k == ord('p'):
            cv2.waitKey(-1)  # wait until any key is pressed

    plt.figure()
    plt.plot(np.arange(0, ile), compression_information[0, :] * 100)
    plt.plot(np.arange(0, ile), compression_information[1, :] * 100)
    plt.plot(np.arange(0, ile), compression_information[2, :] * 100)
    plt.title("File:{}, subsampling={}, divider={}, KeyFrame={} ".format(plik, subsampling, dzielnik,
                                                                         key_frame_counter))
