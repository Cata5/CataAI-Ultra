"use client";

import React, { createContext, useContext, useState, useEffect } from "react";

export type Locale = "en" | "ro";

export const DICTIONARIES = {
  en: {
    app_name: "LLM Maker",
    app_subtitle: "Universal Local AI & Neural Network Workstation",
    overview: "Overview",
    network_studio: "Network Studio",
    training: "Training",
    datasets: "Datasets",
    evaluation: "Evaluation Lab",
    runs: "Runs & History",
    settings: "Settings",
    setup_wizard: "Hardware & Setup Wizard",
    worker: "WORKER",
    workspace: "Workspace",
    switch_workspace: "Switch Workspace",
    create_workspace: "Create Workspace",
    hardware: "Hardware",
    conservative: "Conservative (65%)",
    balanced: "Balanced (80%)",
    performance: "Performance (90%)",
    custom: "Custom",
    safe_margin: "Safe Margin",
    device: "Device",
    vram: "VRAM",
    ram: "RAM",
    precision: "Precision",
    task_family: "Task Family",
    start_training: "Start Training",
    pause_training: "Pause Training",
    stop_training: "Stop Training",
    save_checkpoint: "Save Checkpoint",
    reset_defaults: "Reset Defaults",
    welcome_title: "Welcome to LLM Maker",
    welcome_desc: "A general-purpose, hardware-aware, local-first neural network training and experimentation platform.",
    finish_setup: "Complete Setup & Launch",
    language: "Language",
    romanian: "Română",
    english: "English",
  },
  ro: {
    app_name: "LLM Maker",
    app_subtitle: "Platformă Universală Locală pentru Inteligență Artificială",
    overview: "Prezentare Generală",
    network_studio: "Studio Rețele",
    training: "Antrenare",
    datasets: "Seturi de Date",
    evaluation: "Laborator Evaluare",
    runs: "Rulări și Istoric",
    settings: "Setări Sistem",
    setup_wizard: "Expert Configurare Hardware",
    worker: "PROCES",
    workspace: "Spațiu de Lucru",
    switch_workspace: "Schimbă Spațiul de Lucru",
    create_workspace: "Creează Spațiu Nou",
    hardware: "Hardware",
    conservative: "Conservator (65%)",
    balanced: "Echilibrat (80%)",
    performance: "Performanță (90%)",
    custom: "Personalizat",
    safe_margin: "Marjă de Siguranță",
    device: "Dispozitiv",
    vram: "Memorie VRAM",
    ram: "Memorie RAM",
    precision: "Precizie Calcul",
    task_family: "Familie Obiectiv",
    start_training: "Pornește Antrenarea",
    pause_training: "Pauză Antrenare",
    stop_training: "Oprește Antrenarea",
    save_checkpoint: "Salvează Punct de Control",
    reset_defaults: "Restabilește Implicite",
    welcome_title: "Bun venit în LLM Maker",
    welcome_desc: "Platformă universală locală pentru proiectarea, antrenarea și evaluarea rețelelor neuronale pe orice hardware.",
    finish_setup: "Finalizează Configurarea",
    language: "Limbă",
    romanian: "Română",
    english: "English",
  },
};

type Dictionary = typeof DICTIONARIES.en;

interface I18nContextType {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: keyof Dictionary) => string;
}

const I18nContext = createContext<I18nContextType>({
  locale: "en",
  setLocale: () => {},
  t: (key) => DICTIONARIES.en[key] || key,
});

export const I18nProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [locale, setLocaleState] = useState<Locale>("en");

  useEffect(() => {
    try {
      const saved = localStorage.getItem("llm_maker_locale") as Locale;
      if (saved && (saved === "en" || saved === "ro")) {
        setLocaleState(saved);
      }
    } catch {}
  }, []);

  const setLocale = (newLocale: Locale) => {
    setLocaleState(newLocale);
    try {
      localStorage.setItem("llm_maker_locale", newLocale);
    } catch {}
  };

  const t = (key: keyof Dictionary): string => {
    return DICTIONARIES[locale]?.[key] || DICTIONARIES.en[key] || key;
  };

  return (
    <I18nContext.Provider value={{ locale, setLocale, t }}>
      {children}
    </I18nContext.Provider>
  );
};

export const useTranslation = () => useContext(I18nContext);
