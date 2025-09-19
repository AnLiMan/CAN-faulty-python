#Библиотеки
import numpy as np
import matplotlib.pyplot as plt

plot_font_size = 16 #Размер шрифта графиков

#Класс для генерации вариаций CAN-сигнала
class CANSignalGenerator:
    def __init__(self, bit_rate=500000, sample_rate=10000000):
        self.bit_rate = bit_rate
        self.sample_rate = sample_rate
        self.samples_per_bit = int(sample_rate / bit_rate)

        self.recessive_voltage = 2.5
        self.dominant_high = 3.5
        self.dominant_low = 1.5

    def generate_bit(self, bit_value):
        if bit_value == 0:
            canh = np.full(self.samples_per_bit, self.dominant_high)
            canl = np.full(self.samples_per_bit, self.dominant_low)
        else:
            canh = np.full(self.samples_per_bit, self.recessive_voltage)
            canl = np.full(self.samples_per_bit, self.recessive_voltage)
        return canh, canl

    def generate_id_only_frame(self, can_id):
        bits = [0]
        for i in range(28, -1, -1):
            bits.append((can_id >> i) & 1)
        bits.extend([1, 1, 1, 0, 1, 0, 0, 0])

        canh_samples = np.array([])
        canl_samples = np.array([])

        for bit in bits:
            canh_bit, canl_bit = self.generate_bit(bit)
            canh_samples = np.concatenate([canh_samples, canh_bit])
            canl_samples = np.concatenate([canl_samples, canl_bit])

        return canh_samples, canl_samples

    #Добавление случайных шумов в сигнал
    def add_noise(self, signal, noise_level=0.1):
        noise = np.random.normal(0, noise_level, len(signal))
        return signal + noise

    #Симуляция обрыва провода (open circuit)
    def create_wire_break(self, canh, canl, broken_wire='CANH', break_position=None, break_duration=None):
        canh_fault = canh.copy()
        canl_fault = canl.copy()

        # Установим поломку посреди кадра данных
        if break_position is None:
            break_position = len(canh) // 2
        if break_duration is None:
            break_duration = len(canh) - break_position

        if broken_wire == 'CANH':
            # CAN-H wire broken - signal drops to 0V after break point
            canh_fault[break_position:break_position + break_duration] = 0
            # CAN-L remains normal but may show some artifacts due to missing differential pair

        elif broken_wire == 'CANL':
            # CAN-L wire broken - signal drops to 0V after break point
            canl_fault[break_position:break_position + break_duration] = 0
            # CAN-H remains normal but may show some artifacts

        elif broken_wire == 'BOTH':
            # Both wires broken - complete loss of signal
            canh_fault[break_position:break_position + break_duration] = 0
            canl_fault[break_position:break_position + break_duration] = 0

        return canh_fault, canl_fault

    def add_high_frequency_noise(self, signal, noise_level=0.3, frequency=1000000):
        """Добавить высокочастотный шум от силовых цепей"""
        t = np.arange(len(signal)) / self.sample_rate
        # Генерируем ВЧ шум + случайные всплески
        hf_noise = np.sin(2 * np.pi * frequency * t) * noise_level * 0.5
        random_spikes = np.random.normal(0, noise_level * 0.7, len(signal))
        # Добавляем редкие мощные всплески
        spikes_mask = np.random.random(len(signal)) < 0.02  # 2% вероятность всплеска
        power_spikes = np.where(spikes_mask, np.random.uniform(1.0, 2.0, len(signal)), 0)

        return signal + hf_noise + random_spikes + power_spikes

    def add_slow_edges(self, canh, canl, rise_factor=3.0, fall_factor=3.0):
        """Добавить медленные времена нарастания и спада"""
        canh_slow = canh.copy()
        canl_slow = canl.copy()

        # Находим индексы переходов между состояниями
        diff_canh = np.diff(canh)
        transition_indices = np.where(np.abs(diff_canh) > 0.5)[0]

        for idx in transition_indices:
            # Определяем направление перехода
            if diff_canh[idx] > 0:  # Нарастание (recessive -> dominant)
                # Медленное нарастание для CAN-H
                transition_samples = int(self.samples_per_bit * rise_factor)
                start_idx = max(0, idx - transition_samples // 2)
                end_idx = min(len(canh), idx + transition_samples // 2 + 1)

                # Плавный переход вместо резкого
                transition = np.linspace(canh[idx], canh[idx + 1], end_idx - start_idx)
                canh_slow[start_idx:end_idx] = transition

                # Соответствующий переход для CAN-L
                transition_l = np.linspace(canl[idx], canl[idx + 1], end_idx - start_idx)
                canl_slow[start_idx:end_idx] = transition_l

            else:  # Спад (dominant -> recessive)
                # Медленный спад для CAN-H
                transition_samples = int(self.samples_per_bit * fall_factor)
                start_idx = max(0, idx - transition_samples // 2)
                end_idx = min(len(canh), idx + transition_samples // 2 + 1)

                # Плавный переход вместо резкого
                transition = np.linspace(canh[idx], canh[idx + 1], end_idx - start_idx)
                canh_slow[start_idx:end_idx] = transition

                # Соответствующий переход для CAN-L
                transition_l = np.linspace(canl[idx], canl[idx + 1], end_idx - start_idx)
                canl_slow[start_idx:end_idx] = transition_l

        return canh_slow, canl_slow

    def create_stuck_dominant(self, canh, canl, stuck_line='CANH', start_position=None, duration=None):
        """Создать состояние 'stuck dominant'"""
        canh_stuck = canh.copy()
        canl_stuck = canl.copy()

        if start_position is None:
            start_position = len(canh) // 3
        if duration is None:
            duration = len(canh) // 4

        end_position = start_position + duration

        if stuck_line == 'CANH':
            # CAN-H заклинило в доминантном состоянии (3.5В)
            canh_stuck[start_position:end_position] = self.dominant_high
            # CAN-L может быть либо в доминантном (1.5В), либо плавать
            canl_stuck[start_position:end_position] = self.dominant_low

        elif stuck_line == 'CANL':
            # CAN-L заклинило в доминантном состоянии (1.5В) - менее вероятно, но возможно
            canl_stuck[start_position:end_position] = self.dominant_low
            # CAN-H при этом будет в доминантном (3.5В)
            canh_stuck[start_position:end_position] = self.dominant_high

        return canh_stuck, canl_stuck

    def create_ground_short(self, canh, canl, shorted_line='CANH', short_position=None, short_duration=None):
        """Создать КЗ на массу"""
        canh_short = canh.copy()
        canl_short = canl.copy()

        if short_position is None:
            short_position = len(canh) // 3
        if short_duration is None:
            short_duration = len(canh) // 3

        end_position = short_position + short_duration

        if shorted_line == 'CANH':
            # CAN-H закорочен на массу
            canh_short[short_position:end_position] = 0
            # CAN-L пытается работать, но форма искажена
            canl_short[short_position:end_position] = canl_short[short_position:end_position] * 0.8 + 0.2

        elif shorted_line == 'CANL':
            # CAN-L закорочен на массу
            canl_short[short_position:end_position] = 0
            # CAN-H пытается работать, но форма искажена
            canh_short[short_position:end_position] = canh_short[short_position:end_position] * 0.8 + 0.3

        return canh_short, canl_short

    def create_power_short(self, canh, canl, shorted_line='CANH', short_position=None, short_duration=None,
                           voltage=24.0):
        """Создать КЗ на питание (24V)"""
        canh_short = canh.copy()
        canl_short = canl.copy()

        if short_position is None:
            short_position = len(canh) // 3
        if short_duration is None:
            short_duration = len(canh) // 3

        end_position = short_position + short_duration

        if shorted_line == 'CANH':
            # CAN-H закорочен на +24V
            canh_short[short_position:end_position] = voltage
            # CAN-L пытается работать, но форма сильно искажена
            canl_short[short_position:end_position] = canl_short[short_position:end_position] * 0.6 + 0.4

        elif shorted_line == 'CANL':
            # CAN-L закорочен на +24V
            canl_short[short_position:end_position] = voltage
            # CAN-H пытается работать, но форма сильно искажена
            canh_short[short_position:end_position] = canh_short[short_position:end_position] * 0.6 + 0.3

        return canh_short, canl_short

    def create_canh_canl_short(self, canh, canl, short_position=None, short_duration=None):
        """Создать КЗ между CAN-H и CAN-L"""
        canh_short = canh.copy()
        canl_short = canl.copy()

        if short_position is None:
            short_position = len(canh) // 3
        if short_duration is None:
            short_duration = len(canh) // 3

        end_position = short_position + short_duration

        # При КЗ между CAN-H и CAN-L оба сигнала становятся одинаковыми
        # Усредняем значения двух линий
        average_voltage = (canh[short_position:end_position] + canl[short_position:end_position]) / 2

        canh_short[short_position:end_position] = average_voltage
        canl_short[short_position:end_position] = average_voltage

        return canh_short, canl_short

    def create_no_terminators(self, canh, canl, ringing_amplitude=0.8, damping_factor=0.95):
        """Создать эффект отсутствия терминаторов (звон на фронтах)"""
        canh_ringing = canh.copy()
        canl_ringing = canl.copy()

        # Находим индексы фронтов (переходов между состояниями)
        diff_canh = np.diff(canh)
        transition_indices = np.where(np.abs(diff_canh) > 0.5)[0]

        for idx in transition_indices:
            # Добавляем затухающие колебания после каждого фронта
            ringing_duration = int(self.samples_per_bit * 2)  # Длительность звона

            for i in range(ringing_duration):
                pos = idx + i
                if pos < len(canh):
                    # Затухающая синусоида для эффекта звона
                    decay = damping_factor ** i
                    ringing = ringing_amplitude * np.sin(i * np.pi / 4) * decay

                    canh_ringing[pos] += ringing
                    canl_ringing[pos] -= ringing  # Противоположная фаза для CAN-L

        return canh_ringing, canl_ringing

    def create_too_many_terminators(self, canh, canl, amplitude_factor=0.4):
        """Создать эффект слишком многих терминаторов"""
        canh_low = canh.copy()
        canl_low = canl.copy()

        # Снижаем амплитуду сигналов
        # В recessive состоянии ближе к 2.5V, в dominant - меньший размах
        for i in range(len(canh)):
            if canh[i] > 3.0:  # Dominant high
                canh_low[i] = 2.5 + (canh[i] - 2.5) * amplitude_factor
            else:  # Recessive
                canh_low[i] = 2.5 + (canh[i] - 2.5) * 0.8

            if canl[i] < 2.0:  # Dominant low
                canl_low[i] = 2.5 + (canl[i] - 2.5) * amplitude_factor
            else:  # Recessive
                canl_low[i] = 2.5 + (canl[i] - 2.5) * 0.8

        return canh_low, canl_low

#Отрисовка неиправностей и аномалий
def draw_plot (TIME, CANH, CANL, legend_x = 'Х', legend_y = 'Y', plot_title = ""):
    plt.figure(figsize=(12, 6))
    plt.plot(TIME, CANH, 'b-', label=legend_x, linewidth=2, marker='o', markersize=7, markevery=40)
    plt.plot(TIME, CANL, 'r-', label=legend_y, linewidth=2, marker='x', markersize=7, markevery=40)
    plt.title(plot_title)
    plt.ylabel('Напряжение (В)', fontsize=plot_font_size)
    plt.xlabel('Время (мкс)', fontsize=plot_font_size)
    plt.legend(fontsize=plot_font_size, loc='upper right', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.xticks(fontsize=plot_font_size)  # размер цифр на оси X
    plt.yticks(fontsize=plot_font_size)  # размер цифр на оси Y
    plt.show()

#Основной цикл работы
def main():
    generator = CANSignalGenerator()
    can_id = 0x18FF0104

    # 1. Чистый CAN-сигнал
    canh_clean, canl_clean = generator.generate_id_only_frame(can_id)
    time = np.arange(len(canh_clean)) / generator.sample_rate * 1e6  #Сгенерируем время в микросекундах
    draw_plot(time, canh_clean, canl_clean, 'CAN-H (чистый)', 'CAN-L (чистый)','Чистый CAN-сигнал. ID: 0x18FF0104') #Отрисовка

    # 2. Генерация сигнала с ВЧ шумом
    canh_noisy = generator.add_high_frequency_noise(canh_clean, noise_level=0.18, frequency=1500000)
    canl_noisy = generator.add_high_frequency_noise(canl_clean, noise_level=0.18, frequency=1500000)
    draw_plot(time, canh_noisy, canl_noisy, 'CAN-H (с ВЧ шумом)', 'CAN-L (с ВЧ шумом)',
              'ВЧ шум на CAN-шине - EMI от силовых цепей. ID: 0x18FF0104')  # Отрисовка

    # 3. Генерация сигнала с медленными фронтами
    canh_slow, canl_slow = generator.add_slow_edges(canh_clean, canl_clean, rise_factor=0.8, fall_factor=1.1)
    draw_plot(time, canh_slow, canl_slow, 'CAN-H (медленные фронты)', 'CAN-L (медленные фронты)',
              'Медленные времена нарастания/спада - неисправный трансивер. ID: 0x18FF0104')  # Отрисовка

    #4. Генерация состояния "stuck dominant"
    canh_stuck, canl_stuck = generator.create_stuck_dominant(
        canh_clean, canl_clean,
        stuck_line='CANH',
        start_position=len(canh_clean) // 2,
        duration=len(canh_clean) // 3
    )
    draw_plot(time, canh_stuck, canl_stuck, 'CAN-H (stuck dominant)', 'CAN-L',
              'Состояние "stuck dominant" - заклинивание CAN-H. ID: 0x18FF0104')  # Отрисовка

    # 4. Обрыв провода
    canh_broken, canl_broken = generator.create_wire_break(
        canh_clean, canl_clean,
        broken_wire='CANH',
        break_position=len(canh_clean) // 3,
        break_duration=len(canh_clean) // 2
    )
    # Добавим немного шума для реалистичности
    canh_broken = generator.add_noise(canh_broken, 0.02)
    canl_broken = generator.add_noise(canl_broken, 0.02)
    draw_plot(time, canh_broken, canl_broken, 'CAN-H (обрыв провода)', 'CAN-H (обрыв провода)',
              'Обрыв провода CAN-H. ID: 0x18FF0104')  # Отрисовка

    # 5. Генерация КЗ на массу CAN-H
    canh_gnd_short, canl_gnd_short = generator.create_ground_short(
        canh_clean, canl_clean,
        shorted_line='CANH',
        short_position=len(canh_clean) // 2,
        short_duration=len(canh_clean) // 4
    )
    draw_plot(time, canh_gnd_short, canl_gnd_short, 'CAN-H (КЗ на массу)', 'CAN-L (искаженная форма)',
              'КЗ на массу CAN-H. ID: 0x18FF0104')  # Отрисовка

    # 6. Генерация КЗ на 24V CAN-H
    canh_24v_short, canl_24v_short = generator.create_power_short(
        canh_clean, canl_clean,
        shorted_line='CANH',
        short_position=len(canh_clean) // 2,
        short_duration=len(canh_clean) // 4,
        voltage=24.0
    )

    plt.figure(figsize=(12, 6))
    plt.plot(time, canh_24v_short, 'b-', label='CAN-H (КЗ на 24V)', linewidth=2, marker='o', markersize=7, markevery=40)
    plt.plot(time, canl_24v_short, 'r-', label='CAN-L (сильно искажена)', linewidth=2, marker='x', markersize=7,
             markevery=40)
    plt.axhline(y=24.0, color='orange', linestyle='--', alpha=0.7, label='Напряжение питания (24V)')
    plt.axvspan(time[len(canh_clean) // 2], time[len(canh_clean) // 2 + len(canh_clean) // 4],
                alpha=0.2, color='yellow', label='Область КЗ на 24V')
    plt.title('КЗ на питание 24V (CAN-H) - ID: 0x{:08X}'.format(can_id))
    plt.ylabel('Напряжение (В)', fontsize=plot_font_size)
    plt.xlabel('Время (мкс)', fontsize=plot_font_size)
    plt.legend(fontsize=plot_font_size, loc='upper right', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.ylim(0, 28)
    plt.xticks(fontsize=plot_font_size)
    plt.yticks(fontsize=plot_font_size)
    plt.show()

    # 7. Генерация КЗ между CAN-H и CAN-L
    canh_canl_short, canl_canh_short = generator.create_canh_canl_short(
        canh_clean, canl_clean,
        short_position=len(canh_clean) // 2,
        short_duration=len(canh_clean) // 4
    )

    plt.figure(figsize=(12, 6))
    plt.plot(time, canh_canl_short, 'b-', label='CAN-H (КЗ на CAN-L)', linewidth=2, marker='o', markersize=7,
             markevery=40)
    plt.plot(time, canl_canh_short, 'r-', label='CAN-L (КЗ на CAN-H)', linewidth=2, marker='x', markersize=7,
             markevery=40)
    plt.axvspan(time[len(canh_clean) // 2], time[len(canh_clean) // 2 + len(canh_clean) // 4],
                alpha=0.2, color='purple', label='Область КЗ CAN-H/CAN-L')
    plt.title('КЗ между CAN-H и CAN-L - ID: 0x{:08X}'.format(can_id))
    plt.ylabel('Напряжение (В)', fontsize=plot_font_size)
    plt.xlabel('Время (мкс)', fontsize=plot_font_size)
    plt.legend(fontsize=plot_font_size, loc='upper right', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.ylim(0, 4)
    plt.xticks(fontsize=plot_font_size)
    plt.yticks(fontsize=plot_font_size)
    plt.show()

    # 8. Генерация сигнала без терминаторов
    canh_no_term, canl_no_term = generator.create_no_terminators(
        canh_clean, canl_clean,
        ringing_amplitude=0.6,
        damping_factor=0.92
    )
    draw_plot(time, canh_no_term, canl_no_term, 'CAN-H (без терминаторов)', 'CAN-L (без терминаторов)',
              'Отсутствие терминаторов - "звон" на фронтах. ID: 0x18FF0104')  # Отрисовка

    # 9. Генерация сигнала со слишком многими терминаторами
    canh_many_term, canl_many_term = generator.create_too_many_terminators(
        canh_clean, canl_clean,
        amplitude_factor=0.3
    )
    draw_plot(time, canh_many_term, canl_many_term, 'CAN-H (много терминаторов)', 'CAN-L (много терминаторов)',
              'Слишком много терминаторов - низкая амплитуда. ID: 0x18FF0104')  # Отрисовка

if __name__ == "__main__":
    main()
