#ifndef TQDM_H
#define TQDM_H
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <ios>
#include <iostream>
#include <math.h>
#include <numeric>
#include <string>
#include <unistd.h>
#include <vector>

#define BAR_SHOWN_THRESHOLD 100000000
#define BAR_SHOWN_BYTES_THRESHOLD (100000000 * 6)
#define BAR_SHOWN_BYTES_THRESHOLD_WRITE (100000000 * 3)

class tqdm
{
  public:
    typedef int64_t type;

  private:
    // time, iteration counters and deques for rate calculations
    std::chrono::time_point<std::chrono::system_clock> t_first = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> t_old = std::chrono::system_clock::now();
    int64_t n_old = 0;
    std::vector<double> deq_t;
    std::vector<type> deq_n;
    type nupdates = 0;
    type total_ = 0;
    std::string msg_ = "";
    type period = 1;
    unsigned int smoothing = 50;
    bool use_ema = true;
    float alpha_ema = 0.1;

    std::vector<const char*> bars = {" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"};

    bool in_screen = (system("test $STY") == 0);
    bool in_tmux = (system("test $TMUX") == 0);
    bool is_tty = isatty(1);
    bool use_colors = true;
    bool color_transition = true;
    type width = 40;

    std::string right_pad = "▏";
    std::string label = "";

    void hsv_to_rgb(float h, float s, float v, type& r, type& g, type& b)
    {
        if (s < 1e-6)
        {
            v *= 255.;
            r = v;
            g = v;
            b = v;
        }
        type i = (type)(h * 6.0);
        float f = (h * 6.) - i;
        type p = (type)(255.0 * (v * (1. - s)));
        type q = (type)(255.0 * (v * (1. - s * f)));
        type t = (type)(255.0 * (v * (1. - s * (1. - f))));
        v *= 255;
        i %= 6;
        type vi = (type)v;
        if (i == 0)
        {
            r = vi;
            g = t;
            b = p;
        }
        else if (i == 1)
        {
            r = q;
            g = vi;
            b = p;
        }
        else if (i == 2)
        {
            r = p;
            g = vi;
            b = t;
        }
        else if (i == 3)
        {
            r = p;
            g = q;
            b = vi;
        }
        else if (i == 4)
        {
            r = t;
            g = p;
            b = vi;
        }
        else if (i == 5)
        {
            r = vi;
            g = p;
            b = q;
        }
    }

  public:
    tqdm()
    {
        if (in_screen)
        {
            set_theme_basic();
            color_transition = false;
        }
        else if (in_tmux)
        {
            color_transition = false;
        }
    }

    void reset()
    {
        t_first = std::chrono::system_clock::now();
        t_old = std::chrono::system_clock::now();
        n_old = 0;
        deq_t.clear();
        deq_n.clear();
        period = 1;
        nupdates = 0;
        total_ = 0;
        label = "";
    }

    void set_theme_line() { bars = {"─", "─", "─", "╾", "╾", "╾", "╾", "━", "═"}; }
    void set_theme_circle() { bars = {" ", "◓", "◑", "◒", "◐", "◓", "◑", "◒", "#"}; }
    void set_theme_braille() { bars = {" ", "⡀", "⡄", "⡆", "⡇", "⡏", "⡟", "⡿", "⣿"}; }
    void set_theme_braille_spin() { bars = {" ", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠇", "⠿"}; }
    void set_theme_vertical() { bars = {"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "█"}; }
    void set_theme_basic()
    {
        bars = {" ", " ", " ", " ", " ", " ", " ", " ", "#"};
        right_pad = "|";
    }
    void set_label(std::string label_) { label = label_; }
    void disable_colors()
    {
        color_transition = false;
        use_colors = false;
    }

    void finish()
    {
        progress(total_, total_, msg_);
        printf("\n");
        fflush(stdout);
    }
    void progress(type curr, type tot, std::string msg = "")
    {
        msg_ = msg;
        if (is_tty && (curr % period == 0))
        {
            total_ = tot;
            nupdates++;
            auto now = std::chrono::system_clock::now();
            double dt = ((std::chrono::duration<double>)(now - t_old)).count();
            double dt_tot = ((std::chrono::duration<double>)(now - t_first)).count();
            type dn = curr - n_old;
            n_old = curr;
            t_old = now;
            if (deq_n.size() >= smoothing) deq_n.erase(deq_n.begin());
            if (deq_t.size() >= smoothing) deq_t.erase(deq_t.begin());
            deq_t.push_back(dt);
            deq_n.push_back(dn);

            double avgrate = 0.;
            if (use_ema)
            {
                avgrate = deq_n[0] / deq_t[0];
                for (type i = 1; i < deq_t.size(); i++)
                {
                    double r = 1.0 * deq_n[i] / deq_t[i];
                    avgrate = alpha_ema * r + (1.0 - alpha_ema) * avgrate;
                }
            }
            else
            {
                double dtsum = std::accumulate(deq_t.begin(), deq_t.end(), 0.);
                type dnsum = std::accumulate(deq_n.begin(), deq_n.end(), 0.);
                avgrate = dnsum / dtsum;
            }

            // learn an appropriate period length to avoid spamming stdout
            // and slowing down the loop, shoot for ~25Hz and smooth over 3 seconds
            if (nupdates > 10)
            {
                period = (type)(std::min(std::max((1.0 / 25) * curr / dt_tot, 1.0), 5e5));
                smoothing = 25 * 3;
            }
            double peta = (tot - curr) / avgrate;
            double pct = (double)curr / (tot * 0.01);
            if ((tot - curr) <= period)
            {
                pct = 100.0;
                avgrate = tot / dt_tot;
                curr = tot;
                peta = 0;
            }

            double fills = ((double)curr / tot * width);
            type ifills = (type)fills;

            printf("\015");
            if (use_colors)
            {
                if (color_transition)
                {
                    // red (hue=0) to green (hue=1/3)
                    // type r = 255, g = 255, b = 255;
                    // hsv_to_rgb(0.0 + 0.01 * pct / 3, 0.65, 1.0, r, g, b); // 这里修改颜色
                    // printf("\033[38;2;%ld;%ld;%ldm ", r, g, b);

                    type r = 128, g = 128, b = 128; // 灰色的RGB
                    printf("\033[38;2;%ld;%ld;%ldm ", r, g, b);
                }
                else
                {
                    printf("\033[32m"); // 绿色
                    // printf("\033[33;40;1m");
                }
            }
            std::cout << msg;
            for (type i = 0; i < ifills; i++) std::cout << bars[8];
            if (!in_screen and (curr != tot)) printf("%s", bars[(type)(8.0 * (fills - ifills))]);
            for (type i = 0; i < width - ifills - 1; i++) std::cout << bars[0];
            printf("%s ", right_pad.c_str());
            if (use_colors) printf("\033[1m\033[31m");
            printf("%4.1f%% ", pct);
            if (use_colors) printf("\033[34m");

            std::string unit = "Hz";
            double div = 1.;
            if (avgrate > 1e6)
            {
                unit = "MHz";
                div = 1.0e6;
            }
            else if (avgrate > 1e3)
            {
                unit = "kHz";
                div = 1.0e3;
            }
            printf("[%4ld/%4ld | %3.1f %s | %.0fs<%.0fs] ", curr, tot, avgrate / div, unit.c_str(), dt_tot, peta);
            printf("%s ", label.c_str());
            if (use_colors) printf("\033[0m\033[32m\033[0m\015 "); // 绿色
            // printf("\033[0m\033[33;40;1m\033[0m\015 ");

            if ((tot - curr) > period) fflush(stdout);
        }
    }

    // 定义一个类型别名，用于方便修改类型
    void rgb_to_hsv(float r, float g, float b, float& h, float& s, float& v)
    {
        float min_val = std::min({r, g, b});
        float max_val = std::max({r, g, b});
        float delta = max_val - min_val;

        v = max_val;

        if (max_val > 0)
        {
            s = (delta / max_val);
        }
        else
        {
            // r = g = b = 0
            s = 0;
            h = -1; // undefined
            return;
        }

        if (r == max_val)
        {
            h = (g - b) / delta + ((g < b) ? 6 : 0);
        }
        else if (g == max_val)
        {
            h = (b - r) / delta + 2;
        }
        else
        {
            h = (r - g) / delta + 4;
        }

        h *= 60; // 转换为度数

        // 调整负值
        if (h < 0)
        {
            h += 360;
        }
    }
};

class Bar
{
  private:
    const bool isShown;
    typedef tqdm::type type;
    tqdm* tqdm = nullptr;

  public:
    Bar(bool isShown_ = false) : isShown(isShown_)
    {
        if (isShown)
        {
            tqdm = new class tqdm();
        }
    }

    void progress(type curr, type tot, std::string msg = "")
    {
        if (isShown) tqdm->progress(curr, tot, msg);
    }

    void finish()
    {
        if (isShown) tqdm->finish();
    }
};

#endif

/*
int N = 2000;
    tqdm bar;

    std::cout << "Overhead of loop only:" << std::endl;
    for(int i = 0; i < 100000000; i++) {
        bar.progress(i, 100000000);
    }
    bar.finish();


    std::cout << "Basic:" << std::endl;
    bar.reset();
    bar.set_theme_basic();
    for(int i = 0; i < N; i++) {
        bar.progress(i, N);
        usleep(1000);
    }
    bar.finish();

    std::cout << "Braille:" << std::endl;
    bar.reset();
    bar.set_theme_braille();
    for(int i = 0; i < N; i++) {
        bar.progress(i, N);
        usleep(3000);
    }
    bar.finish();

    std::cout << "Line:" << std::endl;
    bar.reset();
    bar.set_theme_line();
    for(int i = 0; i < N; i++) {
        bar.progress(i, N);
        usleep(3000);
    }
    bar.finish();

    std::cout << "Circles:" << std::endl;
    bar.reset();
    bar.set_theme_circle();
    for(int i = 0; i < N; i++) {
        bar.progress(i, N);
        usleep(3000);
    }
    bar.finish();

    bar.reset();
    std::cout << "Vertical bars:" << std::endl;
    bar.reset();
    bar.set_theme_vertical();
    for(int i = 0; i < N; i++) {
        bar.progress(i, N);
        usleep(3000);
    }
    bar.finish();
*/